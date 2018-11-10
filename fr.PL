#!perl
use strict ;
use feature qw(say);
#use List::Util qw(first sum max shuffle maxstr);
#use List::MoreUtils qw(uniq);
#use File::Spec::Functions;
#use File::Basename qw(dirname basename);
#use Data::Dumper;
#use Scalar::Util qw/reftype/;
#use File::Copy;
#use File::Find;
#use Cwd qw(abs_path);
use strict;
use threads;
use POSIX qw(assert); 
use MIME::Base64 qw( encode_base64 decode_base64);

### global ### 

my $host_name = "algoers.com"; 
my $jd_incr = "jd_incr"; 
my $redis_prefix = qq{redis-cli -h $host_name}; 
##############
&main();


### sub list ###
# main_
sub main()
{
	
	my $jd_xx = &ck_arg_and_ret_jd_xx(\@ARGV); 
	my $b64_fc = &get_jd_xx_content_in_b64($jd_xx);
	my $raw_text = &decode_b64_fc($b64_fc); 
	&print_raw_text_or_binary($raw_text);
}

### sub list ###
sub ck_arg_and_ret_jd_xx(\@)
{
	my $arg_ = shift; 
	my $jd_xx = "NULL"; 
	if (@$arg_ > 0)
	{
		$jd_xx = $arg_->[0]; 
		if ($jd_xx !~ m/jd_\d{1,3}$/)
		{
			my $jd_incr_val = &get_jd_xx_from_incr();
			die "- argv should be jd_xx, less than $jd_incr_val\n"; 
		}
	}
	else
	{
		$jd_xx = &get_jd_xx_from_incr(); 
	}

	my $cml_redis_exist_jd_xx = qq{$redis_prefix exists $jd_xx}; 
	#say $cml_redis_exist_jd_xx; 
	my $res_exist = qx{$cml_redis_exist_jd_xx};

	die "- not exist $jd_xx in redis\n" if $res_exist !~ m/1/; 
	return $jd_xx;
}
sub get_jd_xx_from_incr()
{
	my $res_num = 0;
	my $jd_xx = "jd_$res_num"; 
	my $cml_redis_incr_jd_incr = qq{$redis_prefix get $jd_incr}; 
	$res_num = qx{$cml_redis_incr_jd_incr};
	chomp($res_num);
	my $jd_xx = "jd_$res_num"; 

	return $jd_xx;
}
sub get_jd_xx_content_in_b64($jd_xx)
{
	my $jd_xx = shift; 
	my $res_b64 = "NULL"; 
	my $cml_redis_get_jd_xx = qq{$redis_prefix lrange $jd_xx 0 -1}; 
	$res_b64 = qx{$cml_redis_get_jd_xx};
	#say $res_b64;
	return $res_b64; 
}
sub decode_b64_fc($b64_fc)
{
	my $b64_fc = shift;
	my $raw_text = decode_base64($b64_fc); 
	print $raw_text; 
}

sub print_raw_text_or_binary($raw_text)
{
	my $raw_text = shift; 
	print $raw_text; 
}

