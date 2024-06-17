#!perl
use feature qw(say);

&main(); 

sub main()
{
	### global_ var ###
	my $cpu_threshold = 55; 
	my $sleep_intv = 60;
	my $cml = q!  docker stats --no-stream tfr_wx_v5_0 | awk 'NR==2{print $3}' | cut -d'%' -f1 !;
	my $cml_restart_con = "docker restart tfr_wx_v5_0";
	###################


	#say `$cml`;



	while(1)
	{
		my $cnt = 0;	
		for(my $i=0;$i<5;$i++)
		{
			my $cpu_usage = `$cml`;
			chomp($cpu_usage);

			if ($cpu_usage > $cpu_threshold)
			{
				$cnt++;	
			}
			else
			{
				$i = 0;
				$cnt = 0;
			}

			#say $cpu_usage, " ", $cnt; 
			sleep $sleep_intv;
		}

		# hit 5 times 

		#say "hit restart";
		my $ts = &get_timestamp(1);
		say "[",$ts,"]", $cml_restart_con;
		system($cml_restart_con); 


	}




}


sub get_timestamp($)
{
	my $flag_exact = shift; 

	my $add_zero_if_lt_10 = sub ($num)
	{
		my $num = shift @_;
		if ($num >=10 ){
			return "$num"
		}
		return "0$num";
	};

	my $time_stamp = "NULL";

	my ($second, $minute, $hour, $date, $month, $yearOffset, $dayOfWeek, $dayOfYear, $daylightSavings) = localtime();
	# 17 16 20 24 7 116 3 236 1
	# 2016-08-24 20:16

	my $year = 1900 + $yearOffset;
	$month = $month + 1 ;
	$month = $add_zero_if_lt_10->($month);
	$date = $add_zero_if_lt_10->($date);
	$hour = $add_zero_if_lt_10->($hour);
	$minute = $add_zero_if_lt_10->($minute);
	$time_stamp = "$year$month${date}";  # 20190311

	$time_stamp = "$year$month${date}_$hour$minute" if $flag_exact == 1;

	chomp( $time_stamp );
	return $time_stamp;
}


=pod
#!perl
use strict ;
use feature qw(say);
use List::Util qw(first sum max shuffle maxstr);
#use List::MoreUtils qw(uniq);
use File::Spec::Functions;
use File::Basename qw(dirname basename);
use Data::Dumper;
use Scalar::Util qw/reftype/;
use File::Copy;
use File::Find;
use Cwd qw(abs_path);
use POSIX qw(assert);
use strict;
use threads;
use IO::Handle;


### global var ###
	my $watched_folder = "/root/jd/t/nb";
	$watched_folder = abs_path($ARGV[0]) if @ARGV > 0 && -d $ARGV[0]; 
	$watched_folder = abs_path("./") if @ARGV == 0; 

	my @filter_param_list = qw(h hpp cpp cxx c cc cxx cu cuh); 



	my $edit_log = $watched_folder."/"."bak_script/watch_folder_log.txt";
	my $fn_list = "";
	my $fn_list_new = "";
	my $dst_folder = "/y/git/bwa4cuda"; 
	my $g_cnt = 0;
###

assert(-d $dst_folder); 
assert(-d $watched_folder); 
say "- watch folder: $watched_folder"; 

`rm -rf $edit_log; touch $edit_log`;


if ("__main__" eq "__main__")
{


	find(\&save_fn_list ,  "$watched_folder"); 
	assert($fn_list ne ""); 

	$| = 1; 
	open my $FP_log , ">>$edit_log" or die; 

        while(1)
        {
                find(\&save_fn_list_new ,  $watched_folder);
		assert($fn_list_new ne ""); 

                if ($fn_list_new ne $fn_list)
                {
			my ($new_state, $on_fn) = &analysis_fn_list_get_last_modify($fn_list, $fn_list_new);
			my $timestamp = &get_timestamp();
			
			say $timestamp, "\t", $new_state, "\t" , $on_fn; 

			my $cmd_copy_proc = &gen_process_cmd($new_state, $on_fn);
			say $cmd_copy_proc; 


			my @td_pool = ();
			&td_push(\@td_pool, $cmd_copy_proc);
			&td_join(\@td_pool);

			print $FP_log "$timestamp\t$cmd_copy_proc";

                }

                $fn_list = $fn_list_new;
                $fn_list_new = "";
                sleep 6;
        }
	close $FP_log;
}

### sub list ###
sub gen_process_cmd($new_state, $on_fn)
{
        my ($new_state, $on_fn) = (@_[0], @_[1]);

        my @arr_cmd = ();
        my $port = 10241;
        my $html_url = "algoers.com:$port/";
        my $html_fn = "NULL";


	my $full_fn = $on_fn; 
	my $basename_fn = basename $full_fn;

        if ($new_state ne 'delete')
        {
                        my $e_cmd = qq
                        {
				rm -rf $dst_folder/$basename_fn ;  cp $full_fn $dst_folder/$basename_fn
                        };

                        push @arr_cmd , $e_cmd;
        }
        else
	{
		assert($new_state eq "delete");

		my $e_cmd = qq
		{
			rm -rf $dst_folder/$basename_fn
		};

		push @arr_cmd , $e_cmd;
	}

        return "@arr_cmd";
}

sub analysis_fn_list_get_last_modify($fn_list, $fn_list_new)
{
        my $fn_list = shift;
        my $fn_list_new = shift;

        my $new_state = "modify";   #add,delete
        my $on_fn = "NULL";

        my %hash_fn_ts = ();
        my $cmd_create_hash = q(%hash_fn_ts). qq( = qw($fn_list); );
        eval($cmd_create_hash);


        my %hash_fn_ts_new = ();
        my $cmd_create_hash_new = q(%hash_fn_ts_new). qq( = qw($fn_list_new); );
        eval($cmd_create_hash_new);

        my $len_hash_fn = keys %hash_fn_ts;
        my $len_hash_fn_new = keys %hash_fn_ts_new;


        if ($len_hash_fn_new > $len_hash_fn)
        {
                my $small_ = \%hash_fn_ts;
                my $big_ = \%hash_fn_ts_new;

                $on_fn = &substract_two_hash($big_, $small_);
                $new_state = "add";
        }
        elsif ($len_hash_fn_new < $len_hash_fn)
        {
                my $big_ = \%hash_fn_ts;
                my $small_ = \%hash_fn_ts_new;
                $on_fn = &substract_two_hash($big_, $small_);
                $new_state = "delete";

        }
        else
        {
                assert($len_hash_fn_new == $len_hash_fn);
                my @fn_modified =  sort {-M "$a" <=> -M "$b"} keys %hash_fn_ts_new;
                $on_fn = $fn_modified[0];
                $new_state = "modify";

        }

        assert($new_state eq "add" ||
                $new_state eq "delete" ||
                $new_state eq "modify");

        assert($on_fn ne "NULL");
        return ($new_state, $on_fn);
}

sub substract_two_hash(\%hash_big, \%hash_small)
{
        my ($hash_big_ , $hash_small_) = (@_[0], @_[1]);
        my $ret_del_add_ele = "NULL";
        for my $e (keys (%$hash_big_))
        {
                if (not $e ~~ %$hash_small_)
                {
                        $ret_del_add_ele = $e;
                        last;
                }
        }
        return $ret_del_add_ele;
}

sub if_fn_match_pattern($fn, $ref_filter_param_list)
{
	my $fn = shift;
	my $ref_filter_param_list = shift; 
	my $ret_code = 0;
	# c cc h hpp cu cuh cpp cxx
	
	for my $e_filter (@$ref_filter_param_list)
	{
		if ($fn =~ m/\.$e_filter$/)
		{
			if(-f $fn && $fn !~ m/bak_script/)
			{
				$ret_code = 1; 
			}

			last;
		}
	}

	return $ret_code;
}
sub save_fn_list_new()
{
        my $fn = $File::Find::name;
        if( &if_fn_match_pattern($fn, \@filter_param_list))
        {
                $fn_list_new .= $fn. " " . (stat($fn))[9]."\n";
        }
}

sub save_fn_list()
{
        my $fn = $File::Find::name;
        if( &if_fn_match_pattern($fn, \@filter_param_list))
        {
                $fn_list .= $fn. " " . (stat($fn))[9]."\n";
        }
}



sub add_zero_if_lt_10($num)
{
        my $num = shift @_;
        if ($num >=10 ){
                return "$num"
        }
        return "0$num";
}

sub get_timestamp()
{

# perl get time, for second:
 #perl -e 'print $^T '
#1477400615
my $if_windows = "True";

    my $time_stamp = `date "+%Y%m%d_%H%M"` if $if_windows ne "True";

    if ( $if_windows eq "True"){
        my ($second, $minute, $hour, $date, $month, $yearOffset, $dayOfWeek, $dayOfYear, $daylightSavings) = localtime();
        # 17 16 20 24 7 116 3 236 1
        # 2016-08-24 20:16

        my $year = 1900 + $yearOffset;
        $month = $month + 1 ;
        $month = &add_zero_if_lt_10($month);
        $date = &add_zero_if_lt_10($date);
        $hour = &add_zero_if_lt_10($hour);
        $minute = &add_zero_if_lt_10($minute);
        $time_stamp = "$year$month${date}_${hour}$minute";
    }

    chomp( $time_stamp );
    return $time_stamp;

}



sub td_join($td_pool_)
{
        my $td_pool_ = shift;

        for my $e (@$td_pool_)
        {
                $e->join();
        }
}


sub td_detach($td_pool_)
{
        my $td_pool_ = shift;

        for my $e (@$td_pool_)
        {
                $e->detach();
        }

}


sub td_push()
{
        my $td_pool_ = shift;

        my $cml = shift;

        my $td =  threads->create
        (
                sub
                {
                                        #say $cml;
                                        system($cml);
                }
        );

        my $len = @$td_pool_;
        $td_pool_->[$len] = $td;
}

=cut
