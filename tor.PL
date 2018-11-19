#!perl
# written by jidor at 2018-11-10
# need cpanm:  sudo apt install cpanminus
# 	Redis : cpanm Redis

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
#use threads;
#use POSIX qw(assert); 
use MIME::Base64 qw(encode_base64 decode_base64);
use Redis; 


### global ### 

my $host_name = join "", reverse qw(m o c . s r e o g l a);
#my $host_name = join "", reverse qw(m o c . b a l e w e l);
my $jd_incr = "jd_incr"; 
my $max_file_sz_in_bytes = 10 * 1024 * 1024; # 10M in max

my $max_jd_incr = 256; 
my $r = "NULL";  #redis connection
##############



&main();


### sub list ###
sub main()
{
	my $fc_str_all = &ck_argv_get_fc_str(\@ARGV, $max_file_sz_in_bytes); 
	#say scalar length $fc_str_all; 

	$r = Redis->new( server => "$host_name:6379", debug => 0 );
	my $jd_xx = &incr_jd_incr($jd_incr, $max_jd_incr); 
	say $jd_xx; 

	$r->set($jd_xx, $fc_str_all);
	$r->quit; 
}

sub ck_argv_get_fc_str(\@,$)
{
	my $argv_ = shift; 
	my $max_file_sz_in_bytes = shift;

	die "- please install redis-server by\n\tsudo apt install redis-server\n" if `which redis-cli` eq ""; 

	my $ret_filename = "NULL"; 
	my @ret_fc = (); 
	if (@$argv_ > 0)
	{

		die "- only 1 param needed !\n" if @$argv_ != 1;

		if (-e $argv_->[0])
		{
			open my $FP, "<", $argv_->[0]; 
			@ret_fc = <$FP>; 
			close $FP; 
		}
		else  # an echo string
		{
			if ($^O eq "MSWin32") 
			{
				@ret_fc = ($argv_->[0]); 
			}
			else
			{
				@ret_fc = ($argv_->[0], "\n"); 
			}
		}
	}
	else
	{
		@ret_fc = <STDIN>; 	
	}


	die "- file content or stdin is empty\n" if @ret_fc == 0; 

	my $fc_str_all = join "", @ret_fc;
	if (scalar length $fc_str_all > $max_file_sz_in_bytes)
	{
		die "- file size too big\n"; 	
	}

	return $fc_str_all;
}

sub incr_jd_incr($,$)
{

	my $jd_incr = shift;
	my $max_num = shift;
	my $ret_num = $r->incr($jd_incr);
	if ($ret_num >= $max_num)
	{
		$r->set($jd_incr, 0);
		$ret_num = 0; 
	}

	my $jd_xx = "jd_$ret_num"; 
	return $jd_xx;
}

