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
use POSIX qw(assert); 
use MIME::Base64 qw( encode_base64 decode_base64);

### global ### 

my $host_name = "algoers.com"; 
my $jd_xx = "jd_0"; 
my $b_sz = 1024; 
my $b_num = 20; 

my $redis_prefix = "redis-cli -h $host_name"; 
my $redis_get_all_keys = qq{$redis_prefix keys '*'}; 
my $redis_del_key_jd_xx = qq{$redis_prefix del $jd_xx}; 
my $max_len = $b_num * $b_sz;
##############
&main();


### sub list ###

sub main()
{
	&ck_argv(\@ARGV); 

	my @fc = <>;

	assert(@fc <= $max_len); 
	#say "___". $fc[0]."___";  
	if ($fc[0] =~ m/[^[:print:][:space:]]/)
	{
		$b_sz = int $b_sz / 10;
		$b_num = $b_num * 10;
	}

	&del_redis_key($jd_xx);

	for(my $start = 0; $start <= $max_len; $start += $b_sz)
	{
		my $end = $start + $b_sz -1;
		if ($end >= @fc - 1)
		{
			$end = @fc - 1;
		}; 
		my @e_fc = @fc[$start .. $end]; 
		
		my $e_str_fc = join "", @e_fc; 
		my $cmd_params = &gen_rpush_cmd_params($e_str_fc); 
		&gen_rpush_cmd_and_run($cmd_params); 

		if ($end >= @fc-1)
		{
			last;
		}; 

	}

	#system("cksum *.so"); 
}
### sub list ###
sub ck_argv(\@)
{
	my $argv_ = shift; 
	die "- no need args\n" if @$argv_ > 0;
}

sub gen_rpush_cmd_params(\@e_str_fc)
{
	my $e_str_fc = shift; 
	assert($e_str_fc ne ""); 
	return 	encode_base64 $e_str_fc;  
}

sub del_redis_key($jd_xx)
{
	my $jd_xx = shift;
	chomp($jd_xx); 
	assert($jd_xx ne ""); 

	my @redis_keys = qx{$redis_get_all_keys};

	for my $e_key (@redis_keys)
	{
		chomp($e_key); 

		if ($e_key =~ m/$jd_xx\"{0,1}$/)
		{
			qx($redis_prefix del $jd_xx);
			last; 
		}
	}
}

sub gen_rpush_cmd_and_run()
{
	my $cmd_params = shift;
	my $redis_rpush_cmd = qq{$redis_prefix rpush $jd_xx '$cmd_params'}; 
	#say $redis_rpush_cmd; 
	my $max_each_cli =  length $redis_rpush_cmd; 
	die "- each cli param > 127 kb \n" if $max_each_cli > 126 * 1024; 

	my $res_rpush = qx{$redis_rpush_cmd}; 
	die "$res_rpush " if $res_rpush =~ m/ERR/; 
}






















=pod
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

=cut

