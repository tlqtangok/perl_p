#!perl
use File::Basename qw(dirname basename);
use feature qw(say);
use Data::Dumper;

### global var ###
my $fn_of_file_list = 'txt.txt' if @ARGV == 0;
my $fn_of_file_list = @ARGV[0] if @ARGV != 0;

my @fn = `cat $fn_of_file_list`;
my %hash_suffix = ();
my $DEBUG = 0;

###

for my $e (@fn)
{
	chomp($e);

	if (-T $e)
	{
		(!$DEBUG) && say $e;

		if ($DEBUG)
		{
			$e = basename $e;
			$e =~ s/^.*\.//;
			$hash_suffix{$e}++;
		}
		if (!$DEBUG)
		{

			if ($e =~ m{\.c$|\.cpp$|\.cuh$|\.hpp$|\.cu$|\.h$} )
			{

				say $e; 

			} 

		}
	}


}

($DEBUG) && say Dumper(\%hash_suffix);

