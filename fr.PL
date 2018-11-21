#!perl
# written by jidor at 2018-11-10
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
#use POSIX qw(assert); 
use MIME::Base64 qw( encode_base64 decode_base64);
use Redis; 
#use IO::Compress::Gzip qw(gzip $GzipError); 
use IO::Uncompress::Gunzip qw(gunzip $GunzipError);

### global ### 
my $host_name = join "", reverse qw(m o c . s r e o g l a);
#my $host_name = join "", reverse qw(m o c . b a l e w e l);
my $jd_incr = "jd_incr"; 

my $r = "NULL"; 
my $GZIP_PREFIX = "GZIP_"; 
##############
&main();


### sub list ###
# main_
sub main()
{

	$r = Redis->new( server => "$host_name:6379", debug => 0 );

	my $jd_xx = &ck_arg_and_ret_jd_xx(\@ARGV); 
	my $raw_text = &get_jd_xx_content($jd_xx);
	
	my $gunzip_ret_code = &gunzip_data_if_needed(\$raw_text);

	&print_raw_text_or_binary($raw_text);

	$r->quit;
}

### sub list ###
sub gunzip_data_if_needed(\$)
{
	my $data_src = shift;
	my $data_dst = "";
	my $die_msg =  "- gunzip data failure\n"; 
	my $len_of_gzip_ = length scalar $GZIP_PREFIX;  # 5

	if (substr ($$data_src, 0, $len_of_gzip_) eq $GZIP_PREFIX)
	{
		$$data_src = substr($$data_src, $len_of_gzip_);  
		my $ret_code = gunzip $data_src => \$data_dst or die "$die_msg"; 
		$$data_src = $data_dst;
		return $ret_code; 
	}

	return 1;
}

sub ck_arg_and_ret_jd_xx(\@)
{
	my $arg_ = shift; 
	#die "- please install redis-server by\n\tsudo apt install redis-server\n" if `which redis-cli` eq ""; 

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

	my $res_exist = $r->exists($jd_xx);

	die "- not exist $jd_xx in redis\n" if $res_exist !~ m/1/; 
	return $jd_xx;
}
sub get_jd_xx_from_incr()
{
	my $res_num = 0;
	$res_num = $r->get($jd_incr);  
	my $jd_xx = "jd_$res_num"; 
	return $jd_xx;
}
sub get_jd_xx_content($jd_xx)
{
	my $jd_xx = shift; 
	my $res_raw_text = $r->get($jd_xx); 
	return $res_raw_text; 
}

sub print_raw_text_or_binary($raw_text)
{
	my $raw_text = shift; 
	print $raw_text; 
}

