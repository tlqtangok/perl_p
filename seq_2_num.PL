#!perl 
use strict ;
use feature qw(say);
use List::Util qw(first sum max shuffle maxstr);
#use List::MoreUtils qw(uniq);
use File::Spec::Functions;
use File::Basename qw(dirname basename);
use Data::Dumper;
use Scalar::Util qw/reftype/;
use File::Copy;
use File::Find;
use POSIX qw(assert);
use Cwd qw(abs_path); 
use MIME::Base64 qw( encode_base64 decode_base64); 

### global var ###
my $C_ATGC_to_num = 
{
	"A" => 0,
	"C" => 1,
	"G" => 2,
	"T" => 3,
};

my $C_num_to_ATGC = 
{
	0 => "A",
	1 => "C",
	2 => "G",
	3 => "T",
};
my $C_seq_to_rseq = 
{
	"A" => "T", 
	"G" => "C", 
	"T" => "A", 
	"C" => "G",
};

###

&main();

### sub list ### 
sub main()
{

	

	### main var ###
	my ($seq, $rseq, $seq_raw_in_fq, $seq_to_0_3, $rseq_to_0_3);
	################
	
	my $fq = $ARGV[0]; 
	if ($fq eq "")
	{
		my $error_msg = "- perl test.PL 02233332010001102201000100000001000130020100011310".
		"\n".
		"- perl test.PL ACTCCAAACAGAATCAAACAAAAAAACAAACAGGACCAAACAGTTTTGGA".
		"\n";

		die $error_msg; 	
	}

	if ($fq !~ m/^\d/)
	{
		$seq_raw_in_fq = $fq; 
		$seq = scalar reverse $fq; 
		$rseq = &seq_to_rseq($seq); 
		$seq_to_0_3 = &fun_seq_to_0_3($seq); 
		$rseq_to_0_3 = &fun_seq_to_0_3($rseq); 
		$seq_raw_in_fq = $fq; 
	}

	elsif($fq =~ m/^\d/)
	{
		$seq_to_0_3 = $fq; 
		$seq = &fun_0_3_to_seq($seq_to_0_3);
		$seq_raw_in_fq = scalar reverse $seq; 
	}
	else
	{
		assert(0==1);
	}

	my $rseq = &seq_to_rseq($seq); 
	my $seq_to_0_3 = &fun_seq_to_0_3($seq); 
	my $rseq_to_0_3 = &fun_seq_to_0_3($rseq); 
	&say_all_var($seq_raw_in_fq, $seq, $seq_to_0_3, $rseq, $rseq_to_0_3); 
}

sub fun_seq_to_0_3()
{
	my $seq = shift;
	return join "", (map $C_ATGC_to_num->{$_}, split "",$seq); 
}

sub fun_0_3_to_seq()
{
	my $v_0_3 = shift;
	return join "", (map $C_num_to_ATGC->{$_}, split "",$v_0_3); 
}
sub say_all_var($,$,$,$,$,$)
{
	my ($seq_raw_in_fq, $seq, $seq_to_0_3, $rseq, $rseq_to_0_3) = @_; 
	say "";
	say "- seq in fq";
	say $seq_raw_in_fq; 
	say "\n", "---------", "\n"; 
	say "- seq"; 
	say $seq; 
	say $seq_to_0_3; 
	say "\n", "---------", "\n"; 
	say "- rseq"; 
	say $rseq; 
	say $rseq_to_0_3; 
}
sub seq_to_rseq()
{
	my $seq = shift;
	return join "", (map $C_seq_to_rseq->{$_}, split "",$seq); 
}


