#!perl
use strict ;
use feature qw(say);
#use POSIX qw(assert);
#use List::Util qw(first sum max shuffle maxstr);
#use List::MoreUtils qw(uniq);
#use File::Spec::Functions;
#use File::Basename qw(dirname basename);
#use Data::Dumper;
#use Scalar::Util qw/reftype/;
#use threads;

########################################################################

#say scalar @ARGV; 

die "- need 1 args or 0 args \n" if !(@ARGV == 1 || @ARGV == 0); 


my @fc = ();
my $start_num = 1;

if ( @ARGV == 0)
{
        @fc = <STDIN>;
}
elsif (@ARGV == 1 && -f $ARGV[0])
{
        my $fn = @ARGV[0];
        open my $FP, "<", $fn;
        @fc = <$FP>;
        close $FP;
}
else
{
        die "- not invalid args\n";
}

chomp(@fc);
chomp(@fc);
chomp(@fc);
chomp(@fc);
chomp(@fc);

$ENV{arr_cols_one_line}=join " ", @fc;

print $ENV{arr_cols_one_line}, "\n";


