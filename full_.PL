#perl
use File::Copy;
use feature qw(say); 
use File::Basename qw(dirname basename);
use File::Spec;

my $SP = File::Spec->catfile('', ''); 
use Cwd 'abs_path';

&main();



### sub list ###
sub main()
{
    my $dirname_ = $ARGV[0];
    #print $dirname_;

    my $fn = $ARGV[1];
    #print $fn; 

    my @fc = `type $fn`; 
    for (@fc)
    {
        s/^/$dirname_\\/;
        print $_; 
    }

#system(qq{type $fn| repl "^" "$dirname_\\"} ); 
}



=pod

my $if_windows = 'False';
$if_windows = "True"  if @ENV{WINDIR};

my @ARGV_ = ( @ARGV );

for $file ( @ARGV_)
{

    $_ = $file;
    chomp; 

    my $abs_path_ = abs_path($_);

    if ($if_windows eq "True")
    {
        $abs_path_ =~ s|\/|$SP|g;
    }
    else
    {
    }

    print $abs_path_,"\n";
}
=cut
