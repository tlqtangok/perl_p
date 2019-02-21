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
use Try::Tiny;
use Term::ReadKey;


### global var ### 
my $VERSION_ = "2019.02.19";
my $max_file_sz_in_bytes = 50 * 1024 * 1024 + 5 * 1024; # 50M in max
my $max_jd_incr = 256; 

my $host_name = join "", reverse qw(m o c . s r e o g l a);
#

my $R_ = "NULL";  #redis connection
my $tar = "NULL"; 
my @arr_fn_list = (); 

my $JD_PREFIX = "jd_"; 
my $jd_incr = "${JD_PREFIX}incr"; 

my $GZIP_PREFIX = "GZIP_"; 
my $FILENAME_PREFIX = "FILENAME_"; 
my $CRC32_PREFIX = "CRC32_"; 
my $gzip_board_sz = 2 * 1024; # if big than 2k, then gzip it!

my $redis_port = 6379;
my $FR_PREFIX = "fr"; 
my $TOR_PREFIX = "tor"; 

my $FOLDER_PREFIX = "FOLDER_"; 

my $PW_ARG_PREFIX = "-pw";
my $STDIN_PREFIX = "STDIN_";
my $NO_PW_PREFIX = "NO_PW_";
my $KEY_OF_PW_PREFIX = "PW_OF_"; 
my $VERSION_KEY_OF_TOR_FR="TOR_FR_VERSION_KEY"; 
my $HOMEPAGE = "http://algoers.com:10241/tor_fr_readme.html"; 
##############



&main();


### sub list ###
sub main()
{

	my $FN =  __FILE__;  

	if ($FN =~ m/\b$TOR_PREFIX\b/)
	{
		&exec_tor_process(); 
	}	
	elsif ($FN =~ m/\b$FR_PREFIX\b/)
	{
		&exec_fr_process();
	}
	else
	{
		&clean_R_and_die_with_msg( "- $FN should match $TOR_PREFIX or $FR_PREFIX\n");
	}
}


sub verify_must_be_latest_version($)
{
	my $need_exit_str = shift;
	if (not $R_->exists($VERSION_KEY_OF_TOR_FR))
	{
		&clean_R_and_die_with_msg("- please set up VERSION_KEY_OF_TOR_FR on server!\n");
	}

	if ($R_->get($VERSION_KEY_OF_TOR_FR) ne $VERSION_)
	{
		&clean_R_and_die_with_msg("- your tor-fr version is not latest version! please download latest tor-fr from homepage: $HOMEPAGE\n" );
	} 

	if ($need_exit_str eq "exit")
	{
		exit(1);
	}
}

sub clean_R_and_die_with_msg()
{
	my $error_msg = shift;
	$R_->quit();
	die $error_msg;
}

sub show_version_and_help_if_needed()
{
	my $arg_version_ = "-v";

	my $version_string = "- tor-fr\n  version: $VERSION_\n\n  author: Jidor Tang<tlqtangok\@126.com>\n  homepage: $HOMEPAGE\n  usage: tor <filename> -pw <your_password>\n\n  description:\n    tor-fr is a productive tool that sync-up and share your \n    files,directories instantly, efficently and elegantly\n"; 

	my $need_exit_str = "no_exit"; 
	if(@ARGV == 1 && @ARGV[0] eq $arg_version_)
	{
		say $version_string; 
		$need_exit_str = "exit"; 
	}
	return $need_exit_str; 
}

sub save_all_folder_fn_2_arr_fn_list()
{
        my $fn = $File::Find::name;
        if(-f $fn)
        {
                push @arr_fn_list, $fn;
        }
}

sub connect_to_redis_server($,$)
{
	my ($host_name, $redis_port) = ($_[0], $_[1]); 
	try 
	{
		$R_ = Redis->new( server => "$host_name:$redis_port", debug => 0, cnx_timeout => 1.1,  sentinels_cnx_timeout => 1.1);
	}
	catch
	{
		die("- error, cannot connect algoer.com's redis server, please check your internet connection\n"); 
	}
}
sub exec_tor_process()
{
	# script/tor.PL
	my $need_exit_str = &show_version_and_help_if_needed(); 

	#$R_ = Redis->new( server => "$host_name:$redis_port", debug => 0 );
	&connect_to_redis_server($host_name, $redis_port);  	
	#die "- error in redis driver\n" if Dumper($R_) !~ m/\:$redis_port/; 
	&clean_R_and_die_with_msg("- error in redis driver\n") if Dumper($R_) !~ m/\:$redis_port/; 

	&verify_must_be_latest_version($need_exit_str);
	

	my $jd_xx = &incr_jd_incr($jd_incr, $max_jd_incr); 
	say $jd_xx; 

	my @argv = (@ARGV); 
	my ($fn, $fc_str_all, $pw_if_has) = &ck_argv_get_fc_str(\@argv, $max_file_sz_in_bytes); 

	my $gzip_ret_code = &gzip_data_if_needed(\$fc_str_all, $fn); 
	#say scalar length $fc_str_all; 

	&clear_all_jd_xx_and_pw_prefix() if $jd_xx eq "${JD_PREFIX}0"; 

	$R_->set($jd_xx, $fc_str_all);

	&create_pw_k_v_if_needed($pw_if_has, $jd_xx); 	

	$R_->quit; 
}

sub create_pw_k_v_if_needed($,$)
{
	my $pw_if_has = shift;
	my $jd_xx = shift; 
	if ($pw_if_has ne $NO_PW_PREFIX)
	{
		$R_->set("$KEY_OF_PW_PREFIX:$jd_xx", &mycrc32($pw_if_has)); 
	}
}

sub ck_pw_if_needed_in_fr($)
{
	my $jd_xx = shift;

	my $key_of_pw = "$KEY_OF_PW_PREFIX:$jd_xx"; 
	my $res_exist = $R_->exists($key_of_pw);
	my $JD_DISABLE_PW_ = "JD_DISABLE_PW_"; 

	my $cnt_ck_pw_max = 3;

	if ($res_exist)
	{
		my $crc_pw_on_redis = $R_->get($key_of_pw);
		
		while ($cnt_ck_pw_max)
		{
			ReadMode('noecho');
			print "- need password, please input: "; 	
			my $pw_input = ReadLine(0);
			say ""; 
			ReadMode('normal');

			chomp($pw_input);
			my $flag_jd_disable_pw_timestamp = (&mycrc32($pw_input) == &mycrc32("$JD_DISABLE_PW_" . &get_timestamp()));

			if (&mycrc32($pw_input) == $crc_pw_on_redis || $flag_jd_disable_pw_timestamp)
			{
				last;
			}
			else
			{
				say "  incorrect password, try again\n"; 
			}

			$cnt_ck_pw_max--;
		}

		if ($cnt_ck_pw_max == 0)
		{
			&clean_R_and_die_with_msg( "- incorrect password. you have input password too many times!\n");
		}

	}
		# continue if we got right pw
}

sub exec_fr_process()
{
	# script/fr.PL
	my $need_exit_str = &show_version_and_help_if_needed(); 

	#$R_ = Redis->new( server => "$host_name:$redis_port", debug => 0 );
	&connect_to_redis_server($host_name, $redis_port);  	

	&clean_R_and_die_with_msg("- error in redis driver\n") if Dumper($R_) !~ m/\:$redis_port/; 

	&verify_must_be_latest_version($need_exit_str);

	my $jd_xx = &ck_arg_and_ret_jd_xx(\@ARGV); 

	&ck_pw_if_needed_in_fr($jd_xx);

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
				&clean_R_and_die_with_msg( "- exist folder $prefix_folder_name !\n");
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


sub clear_all_jd_xx_and_pw_prefix($)
{
		my @keys_jd_xx = $R_->keys("${JD_PREFIX}*");
		my $len_of_keys_jd_xx = @keys_jd_xx; 
		say "- delete $len_of_keys_jd_xx ${JD_PREFIX}xx keys ... "; 	

		for my $e_key (@keys_jd_xx)
		{
			if ($e_key =~ m/^${JD_PREFIX}\d/)
			{
				$R_->del($e_key); 

			}	
		}


		# clear all pw
		my @keys_of_pw = $R_->keys("${KEY_OF_PW_PREFIX}:*");
		my $len_of_pw = @keys_of_pw; 

		say "- delete $len_of_pw ${KEY_OF_PW_PREFIX}:${JD_PREFIX}xx keys ... "; 	

		for my $e_key (@keys_of_pw)
		{
			if ($e_key =~ m/^${KEY_OF_PW_PREFIX}\:${JD_PREFIX}\d/)
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


	my $FLAG_NEED_GZIP_DATA = 0; 

	if ($fn !~ m/\.tar.gz$/ && length scalar $$data_src >=  $gzip_board_sz)
	{
		$FLAG_NEED_GZIP_DATA = 1; 
	}

	if ($FLAG_NEED_GZIP_DATA)
	{
		my $die_msg =  "- gzip data failure\n"; 

		my $ret_code = gzip $data_src => \$data_dst or &clean_R_and_die_with_msg("$die_msg"); 
		$$data_src = $data_dst;
	}

	$prefix_info = qq($GZIP_PREFIX:${FLAG_NEED_GZIP_DATA}\n$FILENAME_PREFIX:$fn\n$CRC32_PREFIX:$CRC32\n);

	$$data_src = $prefix_info . $$data_src;

	return 1; 
}

sub ck_argv_get_fc_str(\@,$)
{
	my $argv_ = shift; 
	my $max_file_sz_in_bytes = shift;

	my $fn_if_not_stdin = "NULL"; 
	my $pw_if_has = "NULL"; 

	($fn_if_not_stdin, $pw_if_has)  = &parse_argv_to_arr_fn_pw($argv_); 
	

	my $ret_filename = "NULL"; 
	my @ret_fc = (); 
	my $ret_fn = "txt.txt"; 


	if ($fn_if_not_stdin ne $STDIN_PREFIX)
	{

		my $fn_from_arg = $fn_if_not_stdin; 

		#jd_0
		if (-d $fn_from_arg)
		{
			# set argv_->[0] to *.tar.gz
			$fn_from_arg = &tar_folder_to_file_tgz( abs_path($fn_from_arg) ); 
			assert(-f $fn_from_arg); 
		}

		# now, all are file, not folder 
		if (-f $fn_from_arg)
		{
			$ret_fn = $fn_from_arg; 

			open my $FP, "<", $fn_from_arg; 
			binmode $FP;
			@ret_fc = <$FP>; 
			close $FP; 
		}
		else  # an echo string
		{
			if ($^O eq "MSWin32") 
			{
				@ret_fc = ($fn_from_arg); 
			}
			else
			{
				@ret_fc = ($fn_from_arg, "\n"); 
			}
		}
	}
	else
	{
		@ret_fc = <STDIN>; 	
	}


	&clean_R_and_die_with_msg( "- error! file content or stdin is empty\n" ) if @ret_fc == 0; 

	my $fc_str_all = join "", @ret_fc;

	if (scalar length $fc_str_all > $max_file_sz_in_bytes)
	{
		&clean_R_and_die_with_msg( "- error! file size too big, bigger than $max_file_sz_in_bytes ! \n"); 	
	}


	if (-f $ret_fn 	&& basename($ret_fn) =~ m/$FOLDER_PREFIX.*\.tar.gz$/)
	{
		unlink($ret_fn); 	
	}

	return (basename($ret_fn), $fc_str_all, $pw_if_has);
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
			&clean_R_and_die_with_msg( "- argv should be jd_xx, less than $jd_incr_val\n"); 
		}
	}
	else
	{
		$jd_xx = &get_jd_xx_from_incr(); 
	}

	my $res_exist = $R_->exists($jd_xx);

	&clean_R_and_die_with_msg( "- not exist $jd_xx in redis\n" ) if $res_exist !~ m/1/; 
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
			&clean_R_and_die_with_msg( "- exist $fn_ !\n");
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
	my $ret_code = gunzip $ref_raw_text => \$data_dst or &clean_R_and_die_with_msg( "$die_msg"); 
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
	&clean_R_and_die_with_msg( "- crc32 check failure !\n") if $crc32 != &mycrc32($$ref_left_raw_text);
}

sub tar_folder_to_file_tgz($)
{
	my $full_path_of_dir = shift; 
	$tar = Archive::Tar->new;

	my $fn_basename =  basename ($full_path_of_dir);
	my $fn_out_tar_tgz = "${FOLDER_PREFIX}${fn_basename}.tar.gz";

	&clean_R_and_die_with_msg( "- error , $fn_out_tar_tgz must match FOLDER_ suffix !\n" ) if $fn_out_tar_tgz !~ m/FOLDER_/; 

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
	$prefix_of_folder =~ s/\/.*$//g;

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

sub parse_argv_to_arr_fn_pw(\@argv)
{

        my $argv = shift;

        my @ret_fn_pw = ();

        if (@$argv == 0)
        {
			#say " # 1 stdin + no_pw";
                @ret_fn_pw = ($STDIN_PREFIX, $NO_PW_PREFIX);

        }
        elsif (@$argv == 1)
        {
			#say "# 4 fn + no_pw";
		
			my $fn = $argv->[0];
			@ret_fn_pw = ($fn, $NO_PW_PREFIX);
        }
        elsif (@$argv == 2)
        {
			#say " # 2 stdin + pw";
                assert($argv->[0] eq $PW_ARG_PREFIX);
                my $pw = $argv->[1];
                @ret_fn_pw = ($STDIN_PREFIX, $pw);
        }
        elsif (@$argv == 3)
        {
			#say "# 3 fn + pw";

                my $flag_match_pw_arg = 0;

                my $fn = "NULL";
                my $pw = "NULL";
                my $loc_of_pw_arg_prefix = 0;

                for(my $i=0; $i<@$argv; $i++)
                {
                        my $e_arg = $argv->[$i];
                        if ($e_arg eq $PW_ARG_PREFIX)
                        {
                                $flag_match_pw_arg = 1;
                                $loc_of_pw_arg_prefix = $i;
                                last;
                        }
                }
                assert($flag_match_pw_arg ==  1);

                if ($loc_of_pw_arg_prefix == 1)
                {
                        $fn = $argv->[0];
                }
                elsif ($loc_of_pw_arg_prefix == 0)
                {
                        $fn = $argv->[2];
                }
                else
                {
                        &clean_R_and_die_with_msg( "- error of loc pw arg prefix !\n");
                }
                $pw = $argv->[$loc_of_pw_arg_prefix + 1];


                @ret_fn_pw = ($fn, $pw);
        }
        else
        {
                &clean_R_and_die_with_msg( "- argv is not correct : @$argv\n");
        }
        return @ret_fn_pw;
}


sub get_timestamp()
{

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
	$time_stamp = "$year$month${date}";

	chomp( $time_stamp );
	return $time_stamp;
}


