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
use strict;
use threads;

### global var ###
my $watched_folder = $ENV{nb};
my $fn_list = "";
my $fn_list_new = "";
my $html_folder = q(/root/jd/t/html);
my $g_cnt = 0;
###



find(\&print_it ,  "$watched_folder");

while(1)
{
        find(\&print_it_new ,  $watched_folder);
		
		if ($fn_list_new ne $fn_list)
        {
                say "- meet diff";
                my @fn_modify = &analysis_fn_list_get_last_modify($fn_list_new);
                my $cmd_copy_proc = &gen_copy_cmd(\@fn_modify);
			
				my @td_pool = ();
				&td_push(\@td_pool, $cmd_copy_proc); 
				&td_join(\@td_pool); 

        }

        $fn_list = $fn_list_new;
        $fn_list_new = "";
        sleep 60;
}


### sub list ###
sub gen_copy_cmd(\@fn_modify)
{
        my $fn_modified_ = shift;
        my @arr_cmd = ();

        for my $fn_ipynb (@$fn_modified_)
        {
                my $basename_ipynb = basename $fn_ipynb;
                my $e_cmd = qq
                {
                        rm -rf $html_folder/$basename_ipynb
                        ln -s  $fn_ipynb $html_folder/$basename_ipynb
                        jupyter-nbconvert $html_folder/$basename_ipynb
                };

                push @arr_cmd , $e_cmd;
				last;  # we only proc one !
        }

        #say ( "@arr_cmd" );
        #system( "@arr_cmd" );
        return "@arr_cmd";


}
sub analysis_fn_list_get_last_modify($fn_list)
{
        my $fn_list = shift;
        my %hash_fn_ts = ();
        my $cmd_create_hash = q(%hash_fn_ts). qq( = qw($fn_list); );
        eval($cmd_create_hash);
        #say %hash_fn_ts;
        my @fn_modified =  sort {-M "$a" <=> -M "$b"} keys %hash_fn_ts;
        return @fn_modified[0..1];
}
sub if_fn_match_ipynb($fn)
{
	my $fn = shift;
	my $ret_code = 0; 

	if (-f $fn
		&& $fn =~ m|\.ipynb$|
		&& $fn !~ m|\/.ipynb_checkpoints\/|
		&& $fn !~ m|\/bak_script\/| )
	{
		$ret_code = 1;
	}
	return $ret_code; 
}
sub print_it_new()
{
        my $fn = $File::Find::name;
        if( &if_fn_match_ipynb($fn))
        {
                $fn_list_new .= $fn. " " . (stat($fn))[9]."\n";
        }
}

sub print_it()
{
	my $fn = $File::Find::name;
	if( &if_fn_match_ipynb($fn))
	{
		$fn_list .= $fn. " " . (stat($fn))[9]."\n";
	}
}




#------ no need -------
=pod
sub gen_copy_cmd($fn, $dst_folder)
{
        my $fn = shift;
        my $dst_folder = shift;
        my $fn_r0 = $fn;
        $fn =~ s/.*bgi_gpu_server/$dst_folder/;
        return $fn_r0, $fn;
}

sub gen_size($)
{
        my $fn = shift;
        my $mtime = (stat($fn))[9];
        return $mtime;
}

### perl timestamp ###
sub add_zero_if_lt_10($){
        my $num = shift @_;
        if ($num >=10 ){
                return "$num"
        }
        return "0$num";
}
sub get_timestamp(){

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
=cut

#################

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
					say $cml; 
					system("$cml");
                }
        );

        my $len = @$td_pool_;
        $td_pool_->[$len] = $td;
}


