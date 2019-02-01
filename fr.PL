#!perl
#OK_
# written by jidor at 2018-11-10
# last modified by Jidor Tang <tlqtangok@126.com> at 2019-02-01
#
# need cpanm:  sudo apt install cpanminus
# 	Redis : cpanm Redis
# cp $t/tor.PL $perl_p/tor.PL   &&   cp $t/tor.PL $perl_p/fr.PL
use strict ;
use feature qw(say);
use File::Basename qw(dirname basename);
#use List::Util qw(first sum max shuffle maxstr);
#use List::MoreUtils qw(uniq);
#use File::Spec::Functions;
use Data::Dumper;
#use Scalar::Util qw/reftype/;
#use File::Copy;
#use File::Find;
#use Cwd qw(abs_path);
use strict;
#use threads;
use POSIX qw(assert); 
use MIME::Base64 qw(encode_base64 decode_base64);
use Redis; 
use IO::Compress::Gzip qw(gzip $GzipError); 

use IO::Uncompress::Gunzip qw(gunzip $GunzipError);
use Archive::Tar;
use File::Find;
use File::Path;
use Cwd qw(abs_path cwd);
use File::Spec;

### global var ### 

my $host_name = join "", reverse qw(m o c . s r e o g l a);

my $JD_PREFIX = "jd_"; 
my $jd_incr = "${JD_PREFIX}incr"; 

my $max_file_sz_in_bytes = 50 * 1024 * 1024 + 5 * 1024; # 50M in max

my $max_jd_incr = 256; 
my $R_ = "NULL";  #redis connection
my $GZIP_PREFIX = "GZIP_"; 
my $FILENAME_PREFIX = "FILENAME_"; 
my $CRC32_PREFIX = "CRC32_"; 
my $gzip_board_sz = 2 * 1024; # big than 2k, gzip it!

my $redis_port = 6379;
my $FR_PREFIX = "fr"; 
my $TOR_PREFIX = "tor"; 

our @arr_fn_list = (); 
our $tar = "NULL"; 
our $FOLDER_PREFIX = "FOLDER_"; 

##############



&main();


### sub list ###
sub main()
{
	my $FN =  __FILE__;  

	if($FN =~ m/\b$TOR_PREFIX\b/)
	{
		&exec_tor_process(); 
	}	
	elsif ($FN =~ m/\b$FR_PREFIX\b/)
	{
		&exec_fr_process();
	}
	else
	{
		die "- $FN should match $TOR_PREFIX or $FR_PREFIX\n";
	}
}

sub save_all_folder_fn_2_arr_fn_list()
{
        my $fn = $File::Find::name;
        if(-f $fn)
        {
                #$fn_list .= $fn. " " . (stat($fn))[9]."\n";
                #say $fn; 
                push @arr_fn_list, $fn;
                #return $fn;
        }
}

sub exec_tor_process()
{
	# script/tor.PL
	$R_ = Redis->new( server => "$host_name:$redis_port", debug => 0 );
	die "- error in redis driver\n" if Dumper($R_) !~ m/\:$redis_port/; 

	my $jd_xx = &incr_jd_incr($jd_incr, $max_jd_incr); 
	say $jd_xx; 

	my @argv = (@ARGV); 
	my ($fn, $fc_str_all) = &ck_argv_get_fc_str(\@argv, $max_file_sz_in_bytes); 

	my $gzip_ret_code = &gzip_data_if_needed(\$fc_str_all, $fn); 
	#say scalar length $fc_str_all; 

	&clear_all_jd_xx() if $jd_xx eq "${JD_PREFIX}0"; 

	$R_->set($jd_xx, $fc_str_all);
	$R_->quit; 
}

sub exec_fr_process()
{
	# script/fr.PL
	$R_ = Redis->new( server => "$host_name:$redis_port", debug => 0 );
	die "- error in redis driver\n" if Dumper($R_) !~ m/\:$redis_port/; 

	my $jd_xx = &ck_arg_and_ret_jd_xx(\@ARGV); 
	my $raw_text = &get_jd_xx_content($jd_xx);
	$R_->quit;

	my ($flag_gzip_, $fn_, $crc32,  $left_raw_text) = &parse_prefix_get_gzip_fn($raw_text);

	&prompt_overwrite_if_filename_exists($fn_);

	&gunzip_if_needed($flag_gzip_, $fn_, \$left_raw_text);

	&ensure_crc32_test($crc32, \$left_raw_text); 
	&print_fc_to_fn($fn_, \$left_raw_text);

	$fn_ = &do_untar_to_folder_if_needed($fn_);  #jd_0

	&echo_simplified_fc($fn_); 
}

sub do_untar_to_folder_if_needed($fn_)
{
	my $fn_ = shift;
	my $prefix_folder_name =  "NULL"; 
	if ($fn_ =~ m/^$FOLDER_PREFIX.*tar\.gz$/)
	{
		$prefix_folder_name = $fn_; 	
		$prefix_folder_name =~ s/^$FOLDER_PREFIX(.*)\.tar\.gz/\1/;

		if(-d $prefix_folder_name)
		{
			say "- exists folder $prefix_folder_name, would you like to overwrite? (yes | no)"; 	

			my $ensure_ = <STDIN>;
			chomp($ensure_);
			if ($ensure_ eq "yes" or $ensure_ eq "y" or $ensure_ eq "Y" or $ensure_ eq "YES")
			{
				rmtree($prefix_folder_name);
				assert(not -e $prefix_folder_name);
			}
			else
			{
				say "\n- tarball file file save to $fn_\n"; 
				die "- exist folder $prefix_folder_name !\n";
			}
			say "";
		}
	
		$fn_ = &untar_tgz_file_to_cwd( abs_path($fn_) );
	}

	return $fn_; 
}  

sub mycrc32()
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


sub clear_all_jd_xx($)
{
		my @keys_r = $R_->keys("${JD_PREFIX}*");
		my $len_of_keys = @keys_r; 
		say "- delete $len_of_keys ${JD_PREFIX}xx keys ... "; 	

		for my $e_key (@keys_r)
		{
			if ($e_key =~ m/^${JD_PREFIX}\d/)
			{
				$R_->del($e_key); 
			}	
		}
}

sub gzip_data_if_needed(\$,$)
{
	my $data_src = shift;
	my $fn = shift;  # fn is basename
	my $ret_fn_r0 = "txt.txt"; 
	assert(-f $fn) if $fn ne $ret_fn_r0 && $fn =~ m{\/|\\}; 

	my $data_dst;
	my $prefix_info = "NULL"; 

	my $CRC32 = &mycrc32($$data_src);


	my $FLAG_IS_COMP_BY_Archiver_Tar_IN_TOR = 0; 

	if ($fn !~ m/\.tar.gz$/ && length scalar $$data_src >=  $gzip_board_sz)
	{
		$FLAG_IS_COMP_BY_Archiver_Tar_IN_TOR = 1; 
	}

	if ($FLAG_IS_COMP_BY_Archiver_Tar_IN_TOR)
	{
		my $die_msg =  "- gzip data failure\n"; 

		my $ret_code = gzip $data_src => \$data_dst or die "$die_msg"; 
		$$data_src = $data_dst;
	}

	$prefix_info = qq($GZIP_PREFIX:${FLAG_IS_COMP_BY_Archiver_Tar_IN_TOR}\n$FILENAME_PREFIX:$fn\n$CRC32_PREFIX:$CRC32\n);

	$$data_src = $prefix_info . $$data_src;

=pod
	if (length scalar $$data_src >= $gzip_board_sz)
	{
		my $die_msg =  "- gzip data failure\n"; 

		my $ret_code = gzip $data_src => \$data_dst or die "$die_msg"; 
		$$data_src = $data_dst;
		$prefix_info = "$GZIP_PREFIX:1\n$FILENAME_PREFIX:$fn\n$CRC32_PREFIX:$CRC32\n";
	}
	else
	{
		$prefix_info = "$GZIP_PREFIX:0\n$FILENAME_PREFIX:$fn\n$CRC32_PREFIX:$CRC32\n";
	}

	$$data_src = $prefix_info . $$data_src;
=cut 
	return 1; 
}

sub ck_argv_get_fc_str(\@,$)
{
	my $argv_ = shift; 
	my $max_file_sz_in_bytes = shift;

	#die "- please install redis-server by\n\tsudo apt install redis-server\n" if `which redis-cli` eq ""; 

	my $ret_filename = "NULL"; 
	my @ret_fc = (); 
	my $ret_fn = "txt.txt"; 
	if (@$argv_ > 0)
	{

		die "- only 1 param needed !\n" if @$argv_ != 1;

		#jd_0
		if (-d $argv_->[0])
		{
			# set argv_->[0] to *.tar.gz
			$argv_->[0] = &tar_folder_to_file_tgz( abs_path($argv_->[0]) ); 
			assert(-f $argv_->[0]); 
		}

		# now, all are file, not folder 
		if (-f $argv_->[0])
		{
			$ret_fn = $argv_->[0]; 

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
		die "- file size too big, bigger than $max_file_sz_in_bytes ! \n"; 	
	}


	if (-f $ret_fn 	&& basename($ret_fn) =~ m/$FOLDER_PREFIX.*\.tar.gz$/)
	{
		unlink($ret_fn); 	
	}

	return (basename($ret_fn), $fc_str_all);
}

sub incr_jd_incr($,$)
{

	my $jd_incr = shift;
	my $max_num = shift;

	my $ret_num = $R_->incr($jd_incr);
	if ($ret_num >= $max_num)
	{
		$R_->set($jd_incr, 0);
		$ret_num = 0; 
	}

	my $jd_xx = "${JD_PREFIX}$ret_num"; 
	return $jd_xx;
}

# 2_
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
		if ($jd_xx !~ m/${JD_PREFIX}\d{1,3}$/)
		{
			my $jd_incr_val = &get_jd_xx_from_incr();
			die "- argv should be jd_xx, less than $jd_incr_val\n"; 
		}
	}
	else
	{
		$jd_xx = &get_jd_xx_from_incr(); 
	}

	my $res_exist = $R_->exists($jd_xx);

	die "- not exist $jd_xx in redis\n" if $res_exist !~ m/1/; 
	return $jd_xx;
}
sub get_jd_xx_from_incr()
{
	my $res_num = 0;
	$res_num = $R_->get($jd_incr);  
	my $jd_xx = "${JD_PREFIX}$res_num"; 
	return $jd_xx;
}
sub get_jd_xx_content($jd_xx)
{
	my $jd_xx = shift; 
	my $res_raw_text = $R_->get($jd_xx); 
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

	if (-f $fn)
	{
		say "- file content save to $fn"; 
	}
	elsif(-d $fn)
	{
		say "- folder save to $fn"; 
	}
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
		my $ensure_ = <STDIN>;
		chomp($ensure_);
		if ($ensure_ eq "yes" or $ensure_ eq "y" or $ensure_ eq "Y" or $ensure_ eq "YES")
		{
		}
		else
		{
			die "- exist $fn_ !\n";
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

sub ensure_crc32_test($,\$)
{
	my $crc32 = shift; 
	my $ref_left_raw_text = shift; 
	die "- crc32 check failure !\n" if $crc32 != &mycrc32($$ref_left_raw_text);
}

sub tar_folder_to_file_tgz($)
{
	my $full_path_of_dir = shift; 
	$tar = Archive::Tar->new;

	my $fn_basename =  basename ($full_path_of_dir);
	my $fn_out_tar_tgz = "${FOLDER_PREFIX}${fn_basename}.tar.gz";

	die "- error , $fn_out_tar_tgz must match FOLDER_ suffix !\n" if $fn_out_tar_tgz !~ m/FOLDER_/; 

	assert(-d $full_path_of_dir);
	my $current_dir = cwd(); 
	chdir(dirname $full_path_of_dir); # must change to upper dir !!!
	my $current_dir_new = cwd(); 

	# global var : @arr_fn_list; 	
	assert(@arr_fn_list == 0);  	
	find(\&save_all_folder_fn_2_arr_fn_list, $full_path_of_dir);  # use global @arr_fn_list
	assert(@arr_fn_list != 0); 

	my $sep = File::Spec->catfile('', '');
	map{ $_ = "." . "$sep". File::Spec->abs2rel($_, cwd()); }@arr_fn_list; # to rel path

	$tar->create_archive( $fn_out_tar_tgz, COMPRESS_GZIP, @arr_fn_list );

	assert(-f $fn_out_tar_tgz);

	my $full_fn_out_tar_tgz = $current_dir_new. "$sep". $fn_out_tar_tgz; 


	chdir($current_dir); 

	return $full_fn_out_tar_tgz;
}

sub untar_tgz_file_to_cwd($)
{
	$tar = Archive::Tar->new;
	my $full_fn_tar_tgz = shift; 

	$tar->read($full_fn_tar_tgz);

	my $old_cwd = cwd(); 	
	chdir(dirname $full_fn_tar_tgz); 	

	my @props = ("name"); 
	my $prefix_of_folder = ($tar->list_files(\@props))[0];
	$prefix_of_folder =~ s/\W.*$//g;

	my $fn_basename = basename $full_fn_tar_tgz; 
	#say $fn_basename; 
	assert($fn_basename eq "${FOLDER_PREFIX}${prefix_of_folder}.tar.gz"); 

	$tar->extract(); 

	my $full_path_of_untar_folder = abs_path($prefix_of_folder); 

	assert(-d $full_path_of_untar_folder); 

	#system(qq{find ./$prefix_of_folder -name '*' && rm -rf $prefix_of_folder}); 

	chdir($old_cwd); 
	unlink($full_fn_tar_tgz); 
	return abs_path($full_path_of_untar_folder); 
}

sub test_all()
{
	if (1)
	{
		# do test
		if(0)
		{
			my $full_path_of_dir = "NULL"; 
			$full_path_of_dir = abs_path("."); 	
			#$full_path_of_dir = abs_path($ARGV[0]); 	

			assert(-d $full_path_of_dir);  

			my $full_fn_out_tar_tgz = &tar_folder_to_file_tgz($full_path_of_dir);
			say -s $full_fn_out_tar_tgz; 

			say ""; 

			say $full_fn_out_tar_tgz; 
		}

		if(1)
		{
			my $full_path_of_tgz = "/mnt/hgfs/et/t_/pi/${FOLDER_PREFIX}pi.tar.gz"; 
			my $full_path_of_untar_folder = &untar_tgz_file_to_cwd($full_path_of_tgz);
			say $full_path_of_untar_folder;
		}
	}
}
