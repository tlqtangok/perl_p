#!perl
#BEGIN{@ENV{perl_p_}='/remote/us01home41/linqi/perl_p';push @INC,@ENV{perl_p_};}
BEGIN{push @INC,@ENV{perl_p};}
use feature qw(say);
use Jidor qw(Jidor_Usage subst_dot subst_foreach smart_match);
# -
#  &subst_dot( @arr );                   ;# @arr return to @arr; 
#  &subst_foreach ( $file_name, @arr )   ;# return $cmd is the perl -i.bak -pe ...
#  &smart_match( $tag, @arr_num_or_str ) ;# -1 not match,or $loc_i is return 
# -
#---END Usage 

my $file_name = 'txt.txt'; 
my @arr=qw( 1 2 3 2-2 3.3 4.4-9); 
my $cmd = &subst_foreach( $file_name, @arr) ; 
system($cmd); 

