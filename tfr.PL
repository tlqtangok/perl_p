#!perl
#OK_
# written by jidor at 2018-11-10
# last modified by Jidor Tang <tlqtangok@126.com> at 2019-02-01
#
# need cpanm:  sudo apt install cpanminus
# 	sudo cpan -i  Redis
# 	sudo cpan -i Term::ReadKey
#	sudo cpan -i Term::ProgressBar;

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

use threads;
use threads::shared;

#use POSIX qw(assert); 
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
use arybase; # this is used to make sure exe can be run!!!
use Term::ReadKey;
use Term::ProgressBar;


### global var ### 
my $VERSION_ = "2019.02.24";  # or 2019.02.22:die
my $max_file_sz_in_bytes = 50 * 1024 * 1024 + 5 * 1024; # 50M in max
my $max_jd_incr = 256; 

my $net_speed = 222;  # kb/s
my $net_speed_dl = 222 * 1.0 / 3;  # kb/s

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
my $HOMEPAGE = "http://$host_name:10241/tor_fr_readme.html"; 
my $FLAG_END_REDIS_SET_OR_GET :shared = 0; 
my $show_p_if_time_big_than = 10; 

my $progress_max = 10; 
my $SHARED_PROGRESS :shared = 0; 

my $TOR_SPEED_IP_ts_len_cost_PREFIX = "TOR_SPEED_";
my $FR_SPEED_IP_ts_len_cost_PREFIX = "FR_SPEED_";

my $BIN_NAME_OF_TOOL = "tfr"; 
my $MODULE_NAME_TOR = "t"; 
my $MODULE_NAME_TOR_FULL = "tor"; 
my $MODULE_NAME_FR = "f"; 
my $MODULE_NAME_FR_FULL = "fr"; 

my $MODULE_NAME_VERSION = '-v'; 

##############



&main();


### sub list ###
sub main()
{
	my $arg_error_msg = "- argc must >= 1 and first args can only be one of {t, f or -v}\n"; 

	die $arg_error_msg if @ARGV == 0; 

	my $CALL_SUB_MOUDULE =  $ARGV[0];  

	if ($CALL_SUB_MOUDULE eq $MODULE_NAME_TOR || $CALL_SUB_MOUDULE eq $MODULE_NAME_TOR_FULL)
	{
		&exec_tor_process(); 
	}	
	elsif ($CALL_SUB_MOUDULE eq $MODULE_NAME_FR || $CALL_SUB_MOUDULE eq $MODULE_NAME_FR_FULL)
	{
		&exec_fr_process();
	}
	elsif ($CALL_SUB_MOUDULE eq $MODULE_NAME_VERSION)
	{
		my $need_exit_str = &show_version_and_help_if_needed(); 
		my $current_ip = &connect_to_redis_server($host_name, $redis_port);  	
		#&clean_R_and_die_with_msg("- error in redis driver\n") if Dumper($R_) !~ m/\:$redis_port/; 
		&verify_must_be_latest_version($need_exit_str);
	}
	else
	{
		die $arg_error_msg; 
	}
}

# common__ start
sub clean_R_and_die_with_msg()
{
	my $error_msg = shift;
	$R_->quit();
	die $error_msg;
}

sub show_version_and_help_if_needed()
{
	my $arg_version_ = $MODULE_NAME_VERSION;

	my $need_exit_str = "no_exit"; 
	if(@ARGV == 1 && @ARGV[0] eq $arg_version_)
	{
		my $version_string = "- tor-fr\n  version: $VERSION_\n\n  author: Jidor Tang<tlqtangok\@126.com>\n  homepage: $HOMEPAGE\n\n  usage:\n    $BIN_NAME_OF_TOOL tor <filename> -pw <your_password>\n    $BIN_NAME_OF_TOOL fr jd_xx\n\n  description:\n    tor-fr is a productive tool that sync-up and share your \n    files,directories instantly, efficently and elegantly\n"; 

		say $version_string; 
		$need_exit_str = "exit"; 
	}
	return $need_exit_str; 
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
	};
}

sub verify_must_be_latest_version($)
{
	my $need_exit_str = shift;
	if (not $R_->exists($VERSION_KEY_OF_TOR_FR))
	{
		&clean_R_and_die_with_msg("- please set up VERSION_KEY_OF_TOR_FR on server!\n");
	}


	my $ver_srv = $R_->get($VERSION_KEY_OF_TOR_FR);

	my @ver_pure_die_or_not = split m/\:/, $ver_srv; 
	if (@ver_pure_die_or_not == 1 && $ver_pure_die_or_not[0] ne $VERSION_)
	{
		say ("- warning: tor-fr version is not latest version! please download latest tor-fr from homepage: $HOMEPAGE\n" );
	}
	elsif (@ver_pure_die_or_not == 2 && $ver_pure_die_or_not[0] ne $VERSION_ && $ver_pure_die_or_not[1] eq "die") 
	{
		&clean_R_and_die_with_msg("- error! tor-fr version is not latest version! please download latest tor-fr from homepage: $HOMEPAGE\n" );
	}


	if ($need_exit_str eq "exit")
	{
		exit(1);
	}
}

sub create_progressbar_if_need_long_time($, $)
{

	my $len_of_byte = shift; 
	my $net_speed_ = shift; 
	my $time_second_assume = int ($len_of_byte / 1024 / $net_speed_);
	

	my $flag_show_progress_bar = 0; 

	if ($time_second_assume > $show_p_if_time_big_than)
	{
		$flag_show_progress_bar = 1; 
	}

	
	my $td_progress = "NULL";

	if ($flag_show_progress_bar)
	{

		$| = 1; 
		say ""; 
 		
		my $progressbar = Term::ProgressBar->new ({count => $progress_max, name => 'Progress',ETA=>'linear'});

		my $td_progress_sub =  sub($) 
		{

			while($SHARED_PROGRESS !=  $progress_max)
			{
				$progressbar->update($SHARED_PROGRESS);
				&sleep_exact_seconds(1.0 / 6);
			}
			$progressbar->update($progress_max);
			&sleep_exact_seconds(1.0 / 6);
		};

		$td_progress = threads->create( $td_progress_sub ); 
	}

	return ($td_progress, $flag_show_progress_bar, $time_second_assume); 	
}

sub sleep_exact_seconds($)
{
	my $seconds = shift; 
	select(undef, undef, undef, $seconds); 
}

sub assert_use_die($, $)
{
	my $equ_or_not = shift; 
	my $die_msg = shift; 
	die $die_msg if not $equ_or_not; 
}
# common__ end

# tor___ start
sub exec_tor_process()
{
	# script/tor.PL
	my $need_exit_str = &show_version_and_help_if_needed(); 
	

	my $current_ip = &connect_to_redis_server($host_name, $redis_port);  	

	#&clean_R_and_die_with_msg("- error in redis driver\n") if Dumper($R_) !~ m/\:$redis_port/; 

	&verify_must_be_latest_version($need_exit_str);
	

	my $jd_xx = &incr_jd_incr($jd_incr, $max_jd_incr); 
	say $jd_xx; 


	my @argv = ( @ARGV[1 .. @ARGV - 1] ); 
	my ($fn, $fc_str_all, $pw_if_has) = &ck_tor_arg_get_fc_str(\@argv, $max_file_sz_in_bytes); 

	$SHARED_PROGRESS++; 

	$fn = &gzip_data_if_needed(\$fc_str_all, $fn); 

	$SHARED_PROGRESS += 0.5;
	#say scalar length $fc_str_all; 

	&clear_all_jd_xx_and_pw_prefix() if $jd_xx eq "${JD_PREFIX}0"; 
	$SHARED_PROGRESS += 0.2;


	my ($td_progress_bar, $flag_show_progress_bar, $time_second_assume) = &create_progressbar_if_need_long_time(length scalar $fc_str_all, $net_speed); 

	&redis_set_with_progress_if_needed($jd_xx, $fc_str_all, $fn, $td_progress_bar, $flag_show_progress_bar, $time_second_assume);


	&create_pw_k_v_if_needed($pw_if_has, $jd_xx); 	

	$SHARED_PROGRESS = $progress_max; 
	&sleep_exact_seconds(1.0 / 5); 

	$R_->quit; 
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

sub ck_tor_arg_get_fc_str(\@,$)
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
			&assert_use_die(-f $fn_from_arg, "- $fn_from_arg not file"); 
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

sub gzip_data_if_needed(\$,$)
{
	my $data_src = shift;
	my $fn = shift;  # fn is basename
	my $ret_fn_r0 = "txt.txt"; 
	&assert_use_die(-f $fn, "- should -f $fn") if $fn ne $ret_fn_r0 && $fn =~ m{\/|\\}; 

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

	return $fn; 
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

		# clear all fn 
		my @keys_of_fn = $R_->keys("${FILENAME_PREFIX}:${JD_PREFIX}*");
		my $len_of_fn = @keys_of_fn; 

		say "- delete $len_of_fn ${FILENAME_PREFIX}:${JD_PREFIX}xx keys ... "; 	

		for my $e_key (@keys_of_fn)
		{
			if ($e_key =~ m/^${FILENAME_PREFIX}\:${JD_PREFIX}\d/)
			{
				$R_->del($e_key); 
			}	
		}
}

sub redis_set_with_progress_if_needed($, $, $, $,$)
{
	my ($jd_xx, $fc_str_all, $fn, $td_progress_bar, $flag_show_progress_bar, $time_second_assume) = ($_[0], $_[1], $_[2], $_[3], $_[4], $_[5]);
	my $len_of_byte = length scalar $fc_str_all;
	my $time_r0 = time; 

	if (! $flag_show_progress_bar)
	{
		$R_->set($jd_xx, $fc_str_all);
	}
	else
	{


		#$| =  1; say "";
		$td_progress_bar->detach(); 


		my $td_redis_set = threads->create(
			\&fun_progress_increase_,
			\$FLAG_END_REDIS_SET_OR_GET, 
			$time_second_assume, 
			$progress_max
		); 

		my $td_progress_bar = threads->create(
			sub
			{
				my  ($flag_end_redis_set_or_get_) = ($_[0]); 
				$R_->set($jd_xx, $fc_str_all);
				$$flag_end_redis_set_or_get_ = 1;  
			}
			,
			\$FLAG_END_REDIS_SET_OR_GET); 

		$td_redis_set->join(); 
		$td_progress_bar->join(); 

	}

	$R_->set("$FILENAME_PREFIX:$jd_xx", $fn); 


	my $time_r1 = time; 

	my $cost = $time_r1 - $time_r0; 
	my $ip_ = &get_current_client_ip(); 

	&record_ip_ts_len_cost($TOR_SPEED_IP_ts_len_cost_PREFIX, $ip_, $len_of_byte, $cost);
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
# tor___ end


# fr__ start
sub exec_fr_process()
{
	# script/fr.PL
	my $need_exit_str = &show_version_and_help_if_needed(); 

	&connect_to_redis_server($host_name, $redis_port);  	

	#&clean_R_and_die_with_msg("- error in redis driver\n") if Dumper($R_) !~ m/\:$redis_port/; 

	&verify_must_be_latest_version($need_exit_str);

	my @argv = ( @ARGV[1 .. @ARGV - 1] ); 
	my $jd_xx = &ck_fr_arg_and_ret_jd_xx(\@argv); 

	&ck_pw_if_needed_in_fr($jd_xx);

	my $fn = $R_->get("$FILENAME_PREFIX:$jd_xx"); 
	&prompt_overwrite_if_filename_exists($fn);

	$SHARED_PROGRESS++; 

	my $len_of_byte = $R_->strlen($jd_xx);

	my ($td_progress_bar, $flag_show_progress_bar, $time_second_assume) = &create_progressbar_if_need_long_time($len_of_byte, $net_speed_dl); 


	my $raw_text = &redis_get_with_progress_if_needed($jd_xx, $len_of_byte, $td_progress_bar, $flag_show_progress_bar, $time_second_assume);
	
	$R_->quit;


	my ($flag_gzip_, $fn_, $crc32,  $left_raw_text) = &parse_prefix_get_gzip_fn($raw_text);
	$SHARED_PROGRESS++; 

	&gunzip_if_needed($flag_gzip_, $fn_, \$left_raw_text);
	&ensure_crc32_test($crc32, \$left_raw_text); 
	$SHARED_PROGRESS++;
	
	&print_fc_to_fn($fn_, \$left_raw_text);

	$fn_ = &do_untar_to_folder_if_needed($fn_);  #jd_0
	$SHARED_PROGRESS = $progress_max; 
	&sleep_exact_seconds(1.0 / 5); 

	&echo_simplified_fc($fn_); 
}

sub ck_fr_arg_and_ret_jd_xx(\@)
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
	
	if ($fn_ =~ m/^$FOLDER_PREFIX.*\.tar\.gz$/)
	{
		my $prefix_folder_name = $fn_; 	
		$prefix_folder_name =~ s/^$FOLDER_PREFIX(.*)\.tar\.gz/\1/;

		if(-d $prefix_folder_name)
		{
			say "- exists folder $prefix_folder_name, would you like to overwrite? (yes | no)"; 	

			my $ensure_ = <STDIN>;
			chomp($ensure_);
			if ($ensure_ eq "yes" or $ensure_ eq "y" or $ensure_ eq "Y" or $ensure_ eq "YES")
			{
				rmtree($prefix_folder_name);
				&assert_use_die(not -e $prefix_folder_name, "- should not -e $prefix_folder_name");
			}
			else
			{
				#say "\n- tarball file file save to $fn_\n"; 
				&clean_R_and_die_with_msg( "- exist folder $prefix_folder_name !\n");
			}
			say "";
		}
	}

}

sub redis_get_with_progress_if_needed($, $, $,$,$)
{
	my $jd_xx = shift; 
	my $len_of_byte = shift; 
	my $td_progress_bar = shift; 
	my $flag_show_progress_bar = shift;
	my $time_second_assume = shift;

	my $res_raw_text :shared = "NULL";  

	my $time_r0 = time; 

	if (! $flag_show_progress_bar)
	{
		$res_raw_text =  $R_->get($jd_xx);
	}
	else
	{
		
		#say "";
		$td_progress_bar->detach(); 


		my $td_redis_get = threads->create(
			\&fun_progress_increase_ ,

			\$FLAG_END_REDIS_SET_OR_GET, 
			$time_second_assume,
			$progress_max * 80.0 / 100
		); 

		my $td_progress_bar = threads->create(
			sub
			{
				my  ($flag_end_redis_set_or_get_, $res_raw_text_) = ($_[0], $_[1]); 

				$$res_raw_text_ = $R_->get($jd_xx); 
				&assert_use_die($$res_raw_text_ ne "NULL", "- should res_raw_text_ not NULL"); 
				$$flag_end_redis_set_or_get_ = 1;  
			}
			,
			\$FLAG_END_REDIS_SET_OR_GET, 
			\$res_raw_text); 

		$td_redis_get->join(); 
		$td_progress_bar->join(); 
	}

	&assert_use_die($res_raw_text ne "NULL", "- should res_raw_text not NULL"); 

	my $time_r1 = time; 

	my $cost = $time_r1 - $time_r0; 
	my $ip_ = &get_current_client_ip(); 
	&record_ip_ts_len_cost($FR_SPEED_IP_ts_len_cost_PREFIX, $ip_, $len_of_byte, $cost);

	return $res_raw_text; 
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

sub ensure_crc32_test($,\$)
{
	my $crc32 = shift; 
	my $ref_left_raw_text = shift; 
	&clean_R_and_die_with_msg( "- crc32 check failure !\n") if $crc32 != &mycrc32($$ref_left_raw_text);
}

sub print_fc_to_fn($fn, \$left_raw_text)
{
	my $fn = shift;
	my $ref_left_raw_text = shift;
	open my $FP , ">", $fn or die "- cannot open $fn for read!\n"; 
	binmode $FP;
	print $FP $$ref_left_raw_text; 
	close $FP; 
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
				&assert_use_die(not -e $prefix_folder_name, "- should $prefix_folder_name not exist");
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

# fr__ end 

# mess__ start 
sub save_all_folder_fn_2_arr_fn_list()
{
        my $fn = $File::Find::name;
        if(-f $fn)
        {
                push @arr_fn_list, $fn;
        }
}


sub record_ip_ts_len_cost($,$,$,$)
{
	my ($torfr_prefix, $ip_, $len, $cost) = @_;
	my $ts = &get_timestamp(1); 

	$R_->rpush("${torfr_prefix}${ip_}", "$ts:$len:$cost"); 
}

sub get_current_client_ip()
{
	&sleep_exact_seconds(0.2); 

	my $clients_redis = $R_->client("list"); 
	my $current_redis = (split m/addr./, $clients_redis)[-1]; 
	my $current_client_ip = (split m/\:/, $current_redis)[0]; 
	return $current_client_ip; 
}

sub fun_progress_increase_($,$) 
{
	my  ($flag_end_redis_set_or_get_, $time_second_assume, $progress_max_) = ($_[0], $_[1], $_[2]); 

	my $s_r0 = $SHARED_PROGRESS; 
	my $time_intv = $time_second_assume / ($progress_max_ - $s_r0);

	my $time_split_ = $time_intv * 2; 
	for(my $i=$s_r0; $i<$progress_max_ - 1; $i++)
	{
		my $shared_progress_ = $i * 0.98 + rand(0.44)  ;   # seems more real thing

		if ($$flag_end_redis_set_or_get_) 
		{
			$SHARED_PROGRESS = $progress_max_ - 0.5;
			#&sleep_exact_seconds($time_intv);
			last;
		}

		for(my $ii = 0; $ii < $time_split_; $ii++)
		{
			$SHARED_PROGRESS = $shared_progress_ + $ii / $time_split_;
			&sleep_exact_seconds($time_intv * 1.0 / $time_split_);
		}
	}		
	$SHARED_PROGRESS = $progress_max_ - 0.5;
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



# 2_
### sub list ###
sub get_jd_xx_from_incr()
{
	my $res_num = 0;
	$res_num = $R_->get($jd_incr);  
	my $jd_xx = "${JD_PREFIX}$res_num"; 
	return $jd_xx;
}

# -----------


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


sub parse_right_colon($e_str)
{
	my $e_str = shift;
	my @arr_t = split m/\:/, $e_str;
	die if @arr_t < 2;
	return $arr_t[1]; 
}


sub tar_folder_to_file_tgz($)
{
	my $full_path_of_dir = shift; 
	$tar = Archive::Tar->new;

	my $fn_basename =  basename ($full_path_of_dir);
	my $fn_out_tar_tgz = "${FOLDER_PREFIX}${fn_basename}.tar.gz";

	&clean_R_and_die_with_msg( "- error , $fn_out_tar_tgz must match FOLDER_ suffix !\n" ) if $fn_out_tar_tgz !~ m/FOLDER_/; 

	&assert_use_die(-d $full_path_of_dir,"");
	my $current_dir = cwd(); 
	chdir(dirname $full_path_of_dir); # must change to upper dir !!!
	my $current_dir_new = cwd(); 

	# global var : @arr_fn_list; 	
	&assert_use_die(@arr_fn_list == 0, "");  	
	find(\&save_all_folder_fn_2_arr_fn_list, $full_path_of_dir);  # use global @arr_fn_list
	&assert_use_die(@arr_fn_list != 0, ""); 

	my $sep = File::Spec->catfile('', '');
	map{ $_ = "." . "$sep". File::Spec->abs2rel($_, cwd()); }@arr_fn_list; # to rel path

	$tar->create_archive( $fn_out_tar_tgz, COMPRESS_GZIP, @arr_fn_list );

	&assert_use_die(-f $fn_out_tar_tgz, "");

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
	&assert_use_die($fn_basename eq "${FOLDER_PREFIX}${prefix_of_folder}.tar.gz", ""); 

	$tar->extract(); 

	my $full_path_of_untar_folder = abs_path($prefix_of_folder); 

	&assert_use_die(-d $full_path_of_untar_folder, ""); 

	#system(qq{find ./$prefix_of_folder -name '*' && rm -rf $prefix_of_folder}); 

	chdir($old_cwd); 
	unlink($full_fn_tar_tgz); 
	return abs_path($full_path_of_untar_folder); 
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
                &assert_use_die($argv->[0] eq $PW_ARG_PREFIX, "");
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
                &assert_use_die($flag_match_pw_arg ==  1, "");

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
	$time_stamp = "$year$month${date}";

	$time_stamp = "$year$month${date}_$hour$minute" if $flag_exact == 1;

	chomp( $time_stamp );
	return $time_stamp;
}

=pod
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

			&assert_use_die(-d $full_path_of_dir, "");  

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
=cut

# mess__ end

