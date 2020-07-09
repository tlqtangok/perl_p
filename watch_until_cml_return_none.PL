#!perl
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

#use utf8;
#use Encode; 
#use UTF8; 
#binmode( STDIN,  ':encoding(utf8)' );
#binmode( STDOUT, ':encoding(utf8)' );   
#binmode( STDERR, ':encoding(utf8)' );   



### global var ###

##################

&main();


### sub list ###

	
sub main()
{
	die "- please input cml as string \n" if ! @ARGV; 

	my $watch_cmd_if_no_res_break = $ARGV[0];



	say qq{watch "$watch_cmd_if_no_res_break", util end}; 

	while(1)
	{


		my $cml = `$watch_cmd_if_no_res_break`;



		if ($cml =~ m/^\w/)
		{
			#say $cml; 
			#say "- sleep 5m"; 
			sleep 5*60; 
		}
		else
		{
			sleep 3;
			last;
		}

	}

	say "- done wait ..."; 
}
### sub list ###
