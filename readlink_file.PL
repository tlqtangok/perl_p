
#!perl

BEGIN{push @INC,@ENV{perl_p};}
use feature qw(say);
use Jidor qw(Jidor_Usage subst_dot subst_foreach smart_match);
use File::Basename ;    # get dirname suffix and path
use File::Spec ;
# -
#  &subst_dot( @arr );                   ;# @arr return to @arr; 
#  &subst_foreach ( $file_name, @arr )   ;# return $cmd is the perl -i.bak -pe ...
#  &smart_match( $tag, @arr_num_or_str ) ;# -1 not match,or $loc_i is return 
# -
#---END Usage 

my $file_name = 'txt.txt'; 
my @arr=qw( 1 2 3 2-2 3.3 4.4-9); 
#my $cmd = &subst_foreach( $file_name, @arr) ; 
#system($cmd);
#
################################################ 
my $len=@ARGV; 
my $i=0;
for($i=0;$i < $len; $i++){
print `readlink -f @ARGV[$i]`; 
}
