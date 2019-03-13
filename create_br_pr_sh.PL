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

my $t = "/mnt/share/jd/t/t_bam_20190312"; 
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
	my $FN = $0; 
	say $FN; 
	die "- run\n\tperl $0 <round_num> <each_batch_num> <nct_num>\n" if @ARGV != 3; 


	my @argv = (@ARGV); 
	$max_round = shift @argv;
	$e_b = shift @argv; 
	$nct_num = shift @argv; 

	my $ck_if_var_is_num = sub()
	{
		my $e_var =  shift;
		return ($e_var =~ m/^[\+\-]?\d+$/); 
	};


	assert($ck_if_var_is_num->($e_b));
	assert($ck_if_var_is_num->($max_round));
	assert($ck_if_var_is_num->($nct_num));


	my $fn_bam = "XXX.bam";

	#my @fn_bam_list = qx{ ls /mnt/yangjiaobo_mix/run*_8t/times_1/*/*.soapbwa.sortmark.realn_compress0.bam };	
	my @fn_bam_list = qx{ cat fn_list_bam_from_yjb.txt };
	
	assert(@fn_bam_list>1); 

	my $cnt_round = 0; 
	
	$fn_br_sh = &fn_sh_add_suffix($fn_br_sh);
	$fn_pr_sh = &fn_sh_add_suffix($fn_pr_sh);

	open my $FP_br_sh, ">", $fn_br_sh or die;
	open my $FP_pr_sh, ">", $fn_pr_sh or die;

	while($cnt_round < $max_round)
	{
		#@fn_bam_list = shuffle(@fn_bam_list); 
		my $cnt = 0; 
		my @cmd_br = (); 
		my @cmd_pr = (); 
		for my $fn_bam (@fn_bam_list)
		{
			if (! ($cnt_round < $max_round) ) { last; }

			if (@cmd_br < $e_b)
			{
			}
			else
			{
				say $FP_br_sh "echo begin round $cnt_round\n", q{perl -e 'print time;print qq(\n);'}, "\n", 
				@cmd_br,
				"sleep 2m && free -g\n", "wait\n", q{perl -e 'print time;print qq(\n);'}, "\n", "echo end round $cnt_round\n", q{#_____________________}, "\n"; 

				say $FP_pr_sh "echo begin round $cnt_round\n", q{perl -e 'print time;print qq(\n);'}, "\n", 
				@cmd_pr,
				"sleep 2m && free -g\n", "wait\n", q{perl -e 'print time;print qq(\n);'}, "\n", "echo end round $cnt_round\n", q{#_____________________}, "\n"; 

				$cnt_round++;

				@cmd_br = (); 
				@cmd_pr = (); 
			}
			my $e_cmd = "NULL"; 
			$e_cmd = &create_br_stat($fn_bam);  
			push @cmd_br, $e_cmd; 

			$e_cmd = &create_pr_stat($fn_bam);  
			push @cmd_pr, $e_cmd; 
		}

	}
	close $FP_br_sh; 
	close $FP_pr_sh; 

	say "- save script to $fn_br_sh"; 
	say "- save script to $fn_pr_sh"; 

	say ""; 
	my $logfile = $fn_br_sh . "_" . $fn_pr_sh . "_" . &get_timestamp() .".log";  

	say "- now run :\n\t(sh $fn_br_sh && sh $fn_pr_sh) |& tee $logfile "; 
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


