#!perl
use File::Basename qw(dirname basename);
use feature qw(say); 
use Data::Dumper; 

### global var ###
my $fn_of_file_list = 'txt.txt' if @ARGV == 0;
my $fn_of_file_list = @ARGV[0] if @ARGV != 0;

my @fn = `cat $fn_of_file_list`;
my %hash_suffix = (); 
my $DEBUG = 1; 

### 

for my $e (@fn)
{
        chomp($e);
=pod
        if (-T $e && $e !~ m|\.git[\/\\]|
                && $e !~ m/CMakeFiles[\/\\]/
                && $e !~ m/\.cmake/
                && $e !~ m/\.txt$/i
                && $e !~ m/\.a$/
                && $e !~ m/\.pgm$/
                && $e !~ m/\.xml$/
                && $e !~ m/Makefile$/
                && $e !~ m/\.mk$/
                && $e !~ m/\.ptx$/
                && $e !~ m/\.pdf$/
                && $e !~ m/\.glsl$/
                && $e !~ m/\.ppm$/
                && $e !~ m/\.mtx$/
                && $e !~ m/\.csv$/
                && $e !~ m/\.cpj$/
				&& $e !~ m/\.vert$/
				&& $e !~ m/\.frag$/
				&& $e !~ m/\.dat$/
				&& $e !~ m/\.nvlink$/
				&& $e !~ m/build_readme$/
        )
=cut
        {
			(!$DEBUG) && say $e;

			if ($DEBUG)
			{
				$e =~ s/^.*\.//;
				$hash_suffix{$e}++; 
			}
        }


}

($DEBUG) && say Dumper(\%hash_suffix); 


