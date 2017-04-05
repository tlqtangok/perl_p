#!perl
use strict; 
my $perl_p=@ENV{perl_p};


if (@ARGV==0){
@ARGV[0]=`pwd`;
chomp(@ARGV[0]);

}

my $cmd="perl $perl_p/scp_it.PL"." @ARGV";


my @cmd_ans=`$cmd`; 

map {
s/^.*?\:/ /;
print ;
}@cmd_ans;



