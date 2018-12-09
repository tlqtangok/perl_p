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
my $redis_prefix = qq{redis-cli -h $host_name}; 

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
	$r->quit;

	my ($flag_gzip_, $fn_, $crc32,  $left_raw_text) = &parse_prefix_get_gzip_fn($raw_text);

	&prompt_overwrite_if_filename_exists($fn_);

	&gunzip_if_needed($flag_gzip_, $fn_, \$left_raw_text);

	&ensure_crc32_test($crc32, \$left_raw_text); 
	&print_fc_to_fn($fn_, \$left_raw_text);
	&echo_simplified_fc($fn_); 

}

sub ensure_crc32_test($,\$)
{
	my $crc32 = shift; 
	my $ref_left_raw_text = shift; 
	die "- crc32 check failure !\n" if $crc32 != &mycrc32($$ref_left_raw_text);
}

sub mycrc32 
{
	my ($input, $init_value, $polynomial) = @_;

	$init_value = 0 unless (defined $init_value);
	$polynomial = 0xedb88320 unless (defined $polynomial);

	my @lookup_table;

	for (my $i=0; $i<256; $i++) {
		my $x = $i;
		for (my $j=0; $j<8; $j++) {
			if ($x & 1) {
				$x = ($x >> 1) ^ $polynomial;
			} else {
				$x = $x >> 1;
			}
		}
		push @lookup_table, $x;
	}

	my $crc = $init_value ^ 0xffffffff;

	foreach my $x (unpack ('C*', $input)) {
		$crc = (($crc >> 8) & 0xffffff) ^ $lookup_table[ ($crc ^ $x) & 0xff ];
	}

	$crc = $crc ^ 0xffffffff;

	return $crc;
}


### sub list ###
sub print_fc_to_fn($fn, \$left_raw_text)
{
	my $fn = shift;
	my $ref_left_raw_text = shift;
	open my $FP , ">", $fn or die; 
	binmode $FP;
	print $FP $$ref_left_raw_text; 
	close $FP; 


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


# -----------

sub echo_simplified_fc($)
{
	my $fn = shift;
	if (-T $fn)
	{
		open my $FP, "<", $fn; 
		my @fc = <$FP>;	
		my $max_num_to_echo = 14; 
		if (@fc > $max_num_to_echo)
		{
			my $half_line_num = $max_num_to_echo >> 1; 
			@fc = (@fc[0..$half_line_num], " ...\n" x 1, @fc[-1*$half_line_num..-1]); 

		}
		say @fc; 
		close $FP; 
	}

	say "- file content save to $fn"; 
}

sub gunzip_if_needed($,$,$)
{
	my $flag_gzip_ = shift; 
	my $fn_ = shift;
	my $ref_raw_text = shift;

	if ($flag_gzip_ == 1)
	{
		&do_gunzip($fn_, $ref_raw_text); 
	}
	else
	{
   	
	}
}

sub prompt_overwrite_if_filename_exists($)
{
	my $fn_ = shift;
	my $tmp_file = "txt.txt"; 
	if ($fn_ eq $tmp_file)
	{
	}
	elsif (-f $fn_)
	{
		say "- exists $fn_, would you like to overwrite? (yes | no)"; 	
		my $ensure_ = <>;
		chomp($ensure_);
		if ($ensure_ eq "yes" or $ensure_ eq "y" or $ensure_ eq "Y" or $ensure_ eq "YES")
		{
		}
		else
		{
			die "- exist $fn_ !";
		}
		say "";
	}
}

sub do_gunzip($,$)
{
	my $fn = shift;
	my $ref_raw_text = shift;
	#say "- do_gunzip on raw_text and output to $fn"; 

	my $data_dst = "NULL"; 
	my $die_msg =  "- gunzip data failure\n";
	my $ret_code = gunzip $ref_raw_text => \$data_dst or die "$die_msg"; 
	$$ref_raw_text = $data_dst;

	return $ret_code; 
}

sub parse_prefix_get_gzip_fn($)
{
	my $raw_text = shift;
	my $prefix_ = substr($raw_text, 0, 256);
	my @lines = split m/\n/, $prefix_;
	my $e_0 = $lines[0]; 
	my $e_1 = $lines[1]; 
	my $e_2 = $lines[2]; 

	my $L_0 = length scalar $e_0; 
	my $L_1 = length scalar $e_1; 
	my $L_2 = length scalar $e_2; 


	my $start_offset = $L_0 + $L_1 + $L_2 + 3; 
	my $left_raw_str = substr($raw_text, $start_offset); 
	return (&parse_right_colon($e_0), &parse_right_colon($e_1), &parse_right_colon($e_2), $left_raw_str); 
}


sub parse_right_colon($e_str)
{
	my $e_str = shift;
	my @arr_t = split m/\:/, $e_str;
	die if @arr_t < 2;
	return $arr_t[1]; 
}

