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

	my $fn = "$ENV{t}/stocks_monitor_list.txt"; 

	my @fc = `cat $fn`; 
	die "- please input stock list\n" if ! @ARGV; 

	my @stock_list = (@ARGV);
	for my $e_stock_name (@stock_list)
	{
		&find_each_stock_by_pattern($e_stock_name ,  \@fc);
	}

}


### sub list ###

sub find_each_stock_by_pattern($stock_name, $ref_fc)
{
	my $sn = shift; 
	my $ref_fc_ = shift;
	my $i = 0;
	for($i = @$ref_fc_ -1; $i>=0;$i--)
	{
		my $e_line = $ref_fc_->[$i]; 
		if($e_line =~ m/code.*$sn/)
		{
			last; 
		}
	}

	if($i != 0)
	{
		for(my $j=$i;$j>=0;$j--)
		{
			my $e_line = $ref_fc_->[$j]; 
			if($e_line =~ m/===/)
			{
				my $date_ =  $e_line; 
				print $date_;
				last;
			}
		}

		say $ref_fc_->[$i], $ref_fc_->[$i+1]; 
	}
	else
	{
		say "- no this stock $sn now!"; 
	}
}
