#!perl
# /ssd-cache/temp/jd/t/perl_bin_5_20_1/bin/perl
# written by jidor at 2019-1-19
use strict ;
use feature qw(say);
use List::Util qw(first sum max shuffle maxstr);
#use List::MoreUtils qw(uniq);
#use File::Spec::Functions;
#use File::Basename qw(dirname basename);
use Data::Dumper;
#use Scalar::Util qw/reftype/;
#use File::Copy;
#use File::Find;
#use Cwd qw(abs_path);
use strict;
use threads;
use threads::shared;
use POSIX qw(assert); 
use MIME::Base64 qw( encode_base64 decode_base64);
#use Expect; 
#use Expect::Simple;  # sudo cpan -i Expect::Simple
use strict;
use arybase; # this is used to make sure exe can be run!!!

#use Expect;
#
#

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

use threads;
use threads::shared;

use POSIX qw(assert); 
use MIME::Base64 qw(encode_base64 decode_base64);
#use Redis; 
use IO::Compress::Gzip qw(gzip $GzipError); 

use IO::Uncompress::Gunzip qw(gunzip $GunzipError);
use Archive::Tar;
use File::Find;
use File::Path;
use Cwd qw(abs_path cwd);
use File::Spec;
#use Try::Tiny;
#use JSON; 
#use Term::ReadKey;
#use Term::ProgressBar;

use utf8;
#use Encode; 
#use UTF8; 
#binmode( STDIN,  ':encoding(utf8)' );
#binmode( STDOUT, ':encoding(utf8)' );   
#binmode( STDERR, ':encoding(utf8)' );   



### global var ###
#NULL
##################

&main();


### sub list ###

#main_	
sub main()
{

	my $fn = $ARGV[0]; 

	my $perl_p = $ENV{perl_p};

	# cmd_
	my $cmd_sc = "perl $perl_p/sc.PL"; 
	my $cmd_pe = "perl $perl_p/peval.PL"; 
	my $cmd_tfr = "perl $perl_p/tfr.PL";

	my $res_cml = `cat $fn`;
	my $flag_match_cmd = 0;

	while(1)
	{
		# subs_
		$res_cml =~ s/^\s*sc /sc /i;

		if ( $res_cml =~ m/^sc / )
		{
			$res_cml =~ s/^sc /$cmd_sc /; 
			$flag_match_cmd = 1;
			last;
		}


		$res_cml =~ s/^\s*peval /peval /i;
		$res_cml =~ s/^\s*pe /peval /i;
		if ( $res_cml =~ m/^peval / )
		{
			$res_cml =~ s/^peval /peval '/;
			$res_cml =~ s/$/'/;
			$res_cml =~ s/^peval /$cmd_pe /;
			$flag_match_cmd = 1;
			last;
		}

		$res_cml =~ s/^\s*tor /tor /i;
		if ($res_cml =~ m/^tor /)
		{
			$res_cml = qq ( cat $fn | sed 's/^tor //;' | _CMD_TFR_ t -pw jd );
		}


		$res_cml =~ s/^\s*fr\b/fr /i;
		if ($res_cml =~ m/^fr /)
		{
			chomp ($res_cml);
			$res_cml =~ s/^fr /_CMD_TFR_ f /;
			$res_cml =~ s/\b(\d+)\b/jd_\1/;
			$res_cml .= " -pw jd >/dev/null  2>&1 && cat txt.txt"; 
		}

		if ($res_cml =~ m/_CMD_TFR_ /)
		{
			$res_cml =~ s/_CMD_TFR_ /$cmd_tfr /;
			$flag_match_cmd = 1;
			last;
		}


		last;
	}

	#say $res_cml; 

	my $cmd_default = "cat $fn"; 
	my $res_to_node_js = "NULL"; 
	if ($flag_match_cmd == 1)
	{
		$res_to_node_js = qx{ $res_cml };
	}
	else
	{
		$res_to_node_js = qx {$cmd_default};
	}

	chomp($res_to_node_js);
	print $res_to_node_js; 
}

### sub list ###

