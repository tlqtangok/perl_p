#!perl
use strict ; 
use feature qw(say); 

my $perl_p=@ENV{perl_p}; 
chomp($perl_p);

my $cmd=q(
wpath='sh $perl_p/wpath.sh '
$wpath MY_ARGV_0 |perl -e ' $_=<>;chomp;$m=qq(cmd /c "ecd $_");print $m;'
);

$cmd =~ s/MY_ARGV_0/@ARGV[0]/; 
$cmd =~ s/\$perl_p/$perl_p/; 


#print $cmd; 

my $bat_cmd=` $cmd`; 
system($bat_cmd); 


