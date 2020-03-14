#!perl
# written by jidor at 2019-1-19
use strict ;
use feature qw(say);
use List::Util qw(first sum max shuffle maxstr);
#use List::MoreUtils qw(uniq);
#use File::Spec::Functions;
use File::Basename qw(dirname basename);
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
use JSON; 
use Encode; 
#use URI::Escape;
#use utf8;
#use Term::ReadKey;
#use Term::ProgressBar;

#use Encode; 
#use UTF8; 
#binmode( STDIN,  ':encoding(utf8)' );
#binmode( STDOUT, ':encoding(utf8)' );   
#binmode( STDERR, ':encoding(utf8)' );   



### global var ###

##################
### global var ###
##################

&main();

#my $ip_ =  &pick_one_idle_cluster_ip(); 
#say $ip_; 


### sub list ###
use Digest::MD5;
use Digest::MD5 qw(md5_hex);
use Time::HiRes qw(gettimeofday);

#main_
sub main()
{

	
	die "- no args fn\n" if @ARGV == 0; 

	
	my $fn = $ARGV[0]; 
	my @fc = `cat $fn`; 
	my $perl_p = $ENV{perl_p}; 
	my $sc="perl $perl_p/sc.PL";  

	my $ts = &get_timestamp(); 
	say "=== $ts ==="; 

	for (@fc)
	{
		chomp; 
		next if not m/^\d/; 

		my @arr_code_schead_sctail = split m/\s+/, $_; 
		my @a = @arr_code_schead_sctail; 
		say "code: $arr_code_schead_sctail[0], low: $arr_code_schead_sctail[1], high: $arr_code_schead_sctail[2]"; 
		my @result = qx {$sc $a[1] $a[2]}; 
		say @result; 
	}


}

### sub list ###

sub get_timestamp()
{
	use POSIX qw(strftime);
	use POSIX qw(nice strftime);
	my $today = strftime('%Y-%m-%d %H:%M:%S',localtime());
	return $today; 
}

