#!perl

use File::Basename qw(dirname basename);
use feature qw(say);
use Data::Dumper;

### global var ###
my $fn_of_file_list = @ARGV[0] if @ARGV != 0;
die if ! -e $fn_of_file_list;

open my $FP , "<", $fn_of_file_list;
my @fn = <$FP>; 
close $FP; 

#my @fn = `cat $fn_of_file_list`;
my %hash_suffix = ();
my @filter_param_list = qw(c cpp cuh hpp cu h); 

my $DEBUG = 0;  
###

&main();

### sub list ###

sub main()
{

	if (@ARGV == 2 && $ARGV[1] eq "DEBUG")
	{
		$DEBUG = 1;
	} 
	elsif (@ARGV == 1)
	{
		@filter_param_list = qw(c cpp cuh hpp cu h); 
	}
	elsif (@ARGV >= 2 && $ARGV[1] ne "DEBUG")
	{
		@filter_param_list = @ARGV[1..@ARGV-1];	
	}
	else
	{
		print "- perl filter_src_fn.PL fn_all.txt h c cu\n";
		print "- perl filter_src_fn.PL DEBUG\n";
		die;
	}
	###

	for my $e (@fn)
	{
		chomp($e);

		if (-T $e)
		{
			#(!$DEBUG) && say $e;

			if ($DEBUG)
			{
				$e = basename $e;
				$e =~ s/^.*\.//;
				$hash_suffix{$e}++;
			}
			if (!$DEBUG)
			{
				chomp($e);

				for my $e_filter (@filter_param_list)
				{
					if ($e =~ m/\.$e_filter$/)
					{
						print $e,"\n";
						last;
					}
				}
			}
		}


	}

	#say Dumper(\%hash_suffix);
	($DEBUG) && print ("--------\n",(join "\n",  (sort (keys %hash_suffix))), "\n-----\n\n");


}

