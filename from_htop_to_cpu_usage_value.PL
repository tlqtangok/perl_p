#!perl
use strict ;
use feature qw(say);
use List::Util qw(reduce first sum max shuffle maxstr);
#use List::MoreUtils qw(uniq);
use File::Spec::Functions;
use File::Basename qw(dirname basename);
use Data::Dumper;
use Scalar::Util qw/reftype/;
use File::Copy;
use File::Find;
use Cwd qw(abs_path);
use POSIX qw(assert);
use threads;
use IO::Handle;



### global var ###

###

&main(); 


### sub list ###


sub main()
{

# echo q | htop -C | aha --line-fix | html2text -width 999 | grep -v "F1Help" | grep -v "xml version=" > file.txt



my @fc = `cat file.txt`;

my @cpu_cost = (); 

for my $e (@fc)
{
	next if $e !~ m/\[.*\%\]/;

	chomp($e);

	$e =~ s/\]/]___/g;
	my @t = split m/___/, $e;

	push @cpu_cost , @t;
}

#say @cpu_cost;
map 
{
s/^.*?(\d)/\1/;
} @cpu_cost; 


@cpu_cost =  sort {&get_cpu_id($a) <=> &get_cpu_id($b)} @cpu_cost;

#say join "\n", @cpu_cost;

for my $e (@cpu_cost)
{
my ($e_) = $e =~ /\[.*?(([\d\.]){3,5}\%)\]/;
say &get_cpu_id($e), " => ", $e_; 
}







=pod
	my $fn_fq_list = "fq.list";
	my $ff = "perl $ENV{perl_p}/ff.PL"; 
	my @fc = `cat $fn_fq_list | grep fq.gz| $ff -1`; 

	my $task_all = {}; 
	my $task_cnt = 0;  

	for my $e_l (@fc)
	{
		my $e_cnt = 0; 
		chomp($e_l); 
		my @e_arr_task = split ";", $e_l; 


		for my $e_fq (@e_arr_task)
		{
			$task_all->{"task_$task_cnt"}->{"fn_fq"}->[$e_cnt] = $e_arr_task[$e_cnt];

			$e_cnt++;
		}


		
		my $e_fn_list_sz = reduce { -s $a + -s $b } 0, @e_arr_task;   	
		$task_all->{"task_$task_cnt"}->{"fq_sz"} = $e_fn_list_sz;

		$task_cnt++; 
	}

	say Dumper($task_all); 
=cut

}

sub get_cpu_id()
{
	my $e = shift;
	my ($id_) = $e =~ m/(^\d{1,2})/;
	return $id_; 
}

sub fun_name()
{
	my ($a, $b) = ($_[0], $_[1]);
	say $a; 
	say $b; 
	system("sleep 111s"); 
}

