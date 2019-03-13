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



### global var ###

my $e_b = 2; 
my $max_round = 2;
my $nct_num = "4"; 

my $t = "/home/lush/t/t/br_pr_test_20190309"; 
my $fn_jar = "/home/lush/yangjiaobo/tools/GenomeAnalysisTK-3.8-1/GenomeAnalysisTK.jar"; 
my $fn_ref = "/mnt/hdcancer/db/hg19_chM_male_mask.fa"; 
my $fn_bed = "/mnt/hdcancer/db/for500_region.bed"; 
my $fn_vcf = "/mnt/hdcancer/db/dbsnp_138.hg19.vcf"; 
my $fn_br_sh = "br_sh.sh";
my $fn_pr_sh = "pr_sh.sh";
##################

&main();


### sub list ###

	
sub main()
{
=pod
	my $round_h = {
	"round 0" => {
	flag:0
	br =>{
		t:[]
		 }
	pr =>{
	t:[]
	}

	}; 
=cut

my $round_h = {};
my $fn = $ARGV[0]; 
open my $FP, "<", $fn or die; 
my @fc = <$FP>; 






close $fn;

my $key = ""; 

for(my $i=0;$i<100;$i++)
{
	$key = "round $i"; 
	$round_h->{$key}->{flag} = 0; 
}


chomp(@fc); 

for (my $i=0; $i<@fc; $i++)
{

	if ($fc[$i] =~ m/begin round/)
	{
		$key = $fc[$i]; 
		$round_h->{flag} = 0; 
		($key) = ($fc[$i] =~ /(round \d{1,3})/);
		if($round_h->{$key}->{flag})
		{
			$round_h->{$key}->{pr}->[0] = $fc[$i+1]; 
		}
		else
		{
			$round_h->{$key}->{br}->[0] = $fc[$i+1]; 
		}
	}
	elsif( $fc[$i] =~ m/end round/)
	{
		$key = $fc[$i]; 
		($key) = ($fc[$i] =~ /(round \d{1,3})/);
		#say $key; 
		if($round_h->{$key}->{flag})
		{
			$round_h->{$key}->{pr}->[1] = $fc[$i-1]; 
		}
		else
		{
			$round_h->{$key}->{br}->[1] = $fc[$i-1]; 
		}
		$round_h->{$key}->{flag}++; 
	}

}

#say Dumper($round_h); 

my $br_total_start = 0; 
my $br_total_end = 0; 

my $pr_total_start = 0; 
my $pr_total_end = 0; 

for(my $i=0;$i<100;$i++)
{

	$key = "round $i"; 
	if ($round_h->{$key}->{flag} == 2)
	{
		my $t_point_br = $round_h->{$key}->{br}; 
		my $t_point_pr = $round_h->{$key}->{pr}; 

		my $t_diff_br = $t_point_br->[1] - $t_point_br->[0]; 
		my $t_diff_pr = $t_point_pr->[1] - $t_point_pr->[0];

		if ($i == 0)
		{
			$br_total_start = $t_point_br->[0]; 	
			$pr_total_start = $t_point_pr->[0]; 	
		}	
		else
		{
			$br_total_end = $t_point_br->[1]; 	
			$pr_total_end = $t_point_pr->[1]; 	
		}
		my $rate_br_pr = $t_diff_br / $t_diff_pr; 

		printf "%11s => {br : %4s ; pr : %4s ; rate : %0.2f}\n", $key, $t_diff_br, $t_diff_pr, $rate_br_pr; 
		#printf "$key => {br : $t_diff_br ; pr : $t_diff_pr, rate : $rate_br_pr}"; 
	}
	#$round_h->{$key}->{flag} = 0; 
}


my $br_total_diff = $br_total_end - $br_total_start; 
my $pr_total_diff = $pr_total_end - $pr_total_start; 


printf "\n- {br_total : %4s ; pr_total : %4s}\n", $br_total_diff, $pr_total_diff; 



}


### sub list ###
sub fn_sh_add_suffix($)
{
	my $fn = shift; 
	my $suffix_ = "_${max_round}_x_${e_b}_x_${nct_num}.sh"; 
	$fn =~ s/\.sh$/$suffix_/;
	return $fn; 
}

sub create_br_stat($,$)
{
	my $fn_bam = shift; 
	chomp($fn_bam); 

	my $b_bam  = basename $fn_bam; 
	$b_bam =~ s/\.bam$/.grp/;

	my $fn_grp = "${b_bam}"; 
	return qq{\tjava -Djava.io.tmpdir=$t -jar $fn_jar -R $fn_ref -T BaseRecalibrator -nct $nct_num  -L $fn_bed -I $fn_bam -knownSites  $fn_vcf -o $t/$fn_grp &\n}; 
}

sub create_pr_stat($,$)
{
	my $fn_bam = shift; 
	chomp($fn_bam); 

	my $b_bam  = basename $fn_bam; 
	$b_bam =~ s/\.bam$/.grp/;

	my $fn_bam_new  = basename $fn_bam; 
	$fn_bam_new =~ s/\.bam$/_new.bam/;

	my $fn_grp = "${b_bam}"; 
	return qq{\tjava -Djava.io.tmpdir=$t -jar $fn_jar -R $fn_ref -T PrintReads -nct $nct_num -filterNoBases -I $fn_bam -BQSR $t/$fn_grp -o $t/$fn_bam_new &\n}; 

}

sub assert_use_die($, $)
{
	my $equ_or_not = shift;
	my $die_msg = shift; 

        $equ_or_not = int($equ_or_not); 	
	die $die_msg if ! $equ_or_not; 
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
	$time_stamp = "$year$month${date}_${hour}$minute";

	chomp( $time_stamp );
	return $time_stamp;
}

sub do_test_input_sleep()
{
	print "- input your userid: "; 
	my $userid = <STDIN>; 
	sleep .1; 
	print "- input your passwd: "; 
	my $passwd = <STDIN>; 

	chomp($userid); 
	chomp($passwd); 
	#say "";

	say $userid, ":" , $passwd; 
}
sub e_exec()
{
	my $arg = shift; 
	say $arg; 
	system( qq{echo $arg >> 1.txt} ); 
}
