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
use Try::Tiny;
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
    my $curdir = cwd(); 
    die if ! -f "${curdir}/lsh.PL";

    system(
        qq{
        curl http://algoers.com:10241/tor_fr/linux/tfr > /tmp/tfr_
        sudo cp /tmp/tfr_  /bin/
        rm /tmp/tfr_
        base64 -d b64.bash_aliases |grep -v ssh_ > .bash_aliases
        cp ./vimrc $ENV{HOME}/
        }
    ); 

    my @fn_bashrc = `cat .bashrc`;
    my $fn_new_home_bashrc = "$ENV{HOME}/.bashrc";
    open my $FP_home_bashrc , ">>", $fn_new_home_bashrc;

    my $flag_output = 0; 
    for my $e_l (@fn_bashrc) 
    {
        if ($e_l =~ m/jd export/)
        {$flag_output = 1;}
        if ($flag_output)
        {
             print $FP_home_bashrc $e_l;
        }
    }
    close $FP_home_bashrc;

}


### sub list ###
