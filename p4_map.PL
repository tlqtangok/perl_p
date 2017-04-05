#!perl

my $pwd = `pwd`; 
chmod ( $pwd ); 


my $P4CLIENT= @ENV{P4CLIENT}; 

die "- P4CLIENT not equ linqi_tmp_ \n" if $P4CLIENT ne "linqi_tmp_" ; 
die "- PWD not match linqi_tmp_ \n" if (!($pwd =~ m/\/linqi_tmp_\//)) ; 


my $pwd_tail = $pwd ; 
$pwd_tail =~ s/^.*(\/linqi_tmp_\/)/\1/; 

my $p4_client_o_matcher = $pwd_tail ; 
$p4_client_o_matcher =~ s/\/linqi_tmp_(\/.*?\/).*$/\1/;

my @arr_p4_client_o = ` p4 client -o | grep $p4_client_o_matcher `; 
chmod ( @arr_p4_client_o[0] ); 

my $p4_match_view = @arr_p4_client_o[0] ; 

my $web_p4_depot_addr = 





