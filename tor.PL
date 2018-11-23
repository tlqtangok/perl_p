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
use IO::Compress::Gzip qw(gzip $GzipError); 

#use IO::Uncompress::Gunzip qw(gunzip $GunzipError);


### global ### 

my $host_name = join "", reverse qw(m o c . s r e o g l a);
#my $host_name = join "", reverse qw(m o c . b a l e w e l);
my $jd_incr = "jd_incr"; 
my $max_file_sz_in_bytes = 50 * 1024 * 1024 + 5 * 1024; # 50M in max

my $max_jd_incr = 256; 
my $r = "NULL";  #redis connection
my $GZIP_PREFIX = "GZIP_"; 
my $gzip_board_sz = 2 * 1024; # big than 2k, gzip it!
##############



&main();


### sub list ###
sub main()
{
	my $fc_str_all = &ck_argv_get_fc_str(\@ARGV, $max_file_sz_in_bytes); 

	my $gzip_ret_code = &gzip_data_if_needed(\$fc_str_all); 
	#say scalar length $fc_str_all; 

	$r = Redis->new( server => "$host_name:6379", debug => 0 );
	my $jd_xx = &incr_jd_incr($jd_incr, $max_jd_incr); 
	say $jd_xx; 


	&clear_all_jd_xx() if $jd_xx eq "jd_0"; 

	$r->set($jd_xx, $fc_str_all);
	$r->quit; 
}

sub clear_all_jd_xx($)
{
		my @keys_r = $r->keys("jd_*");
		my $len_of_keys = @keys_r; 
		say "- delete $len_of_keys jd_xx keys ... "; 	

		for my $e_key (@keys_r)
		{
			if ($e_key =~ m/^jd_\d/)
			{
				$r->del($e_key); 
			}	
		}
}

sub gzip_data_if_needed(\$)
{
	my $data_src = shift;
	my $data_dst;

	if (length scalar $$data_src >= $gzip_board_sz)
	{
		my $die_msg =  "- gzip data failure\n"; 

		my $ret_code = gzip $data_src => \$data_dst or die "$die_msg"; 
		$$data_src = "$GZIP_PREFIX".$data_dst;
		return $ret_code; 
	}
	return 1; 
}

sub ck_argv_get_fc_str(\@,$)
{
	my $argv_ = shift; 
	my $max_file_sz_in_bytes = shift;

	#die "- please install redis-server by\n\tsudo apt install redis-server\n" if `which redis-cli` eq ""; 

	my $ret_filename = "NULL"; 
	my @ret_fc = (); 
	if (@$argv_ > 0)
	{

		die "- only 1 param needed !\n" if @$argv_ != 1;

		if (-e $argv_->[0])
		{
			open my $FP, "<", $argv_->[0]; 
			binmode $FP;
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

=pod
	my $fn_fc_str_all = "fn_fc_str_all"; 

	open my $fp_fc_str_all, ">",$fn_fc_str_all;  
	binmode $fp_fc_str_all; 
	print $fp_fc_str_all $fc_str_all;
	close $fp_fc_str_all; 
=cut

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

