#!perl
use feature qw(say);

@template=qw(
HS234_base
HS236_base
HS34
HS34_base
HS36
HS36_base
HS434_base
HS436_base

em4_ecc
em4_parity
em4_rtos
em4_sensor
em6_gp

ARC710D      
ARC710D_AHB  
ARC725D      
ARC725D_AHB  
ARC750D      
ARC750D_AHB  
ARC770D      
ARC770D_AHB  
ARC770D_AXI

625x2        
ARC601       
ARC601_AHB   
ARC605       
ARC610D      
ARC610D_AHB  
ARC625D      
ARC625D_AHB
AS210
AS211SFX
AS221BD


);
$ARChitect2_cl='ARChitect2 -cl  -libraries %IP_HOME%\* -projectpath C:\ARC\Projects\p_%template%\  -argument_file C:\ARC\Projects\build_configuration\build_configuration_%template%.txt -overwrite';

sub out_build_bat{
	$HS_iphome='set IP_HOME=C:\Jidor_Software\synopsys\ARC_HS_Series_IP_library\ARC_HS_IP-Libraries'; 
$em_iphome= 'set IP_HOME=C:\Jidor_Software\synopsys\ARC_EM_Series_IP_library\ARC_EM_IP-Libraries';
$_700_iphome='set IP_HOME=C:\Jidor_Software\synopsys\ARC_700_Series_IP_library\ARC700_IP-Libraries\ARC700_IP-Libraries';
$_600_iphome='set IP_HOME=C:\Jidor_Software\synopsys\ARC_600_Series_IP_library\ARC600_IP-Libraries';
$_=@_[0]; # Get the input out_build_bat( args = $_ ); 
my $IP_HOME='';
$fn="$_".'.bat';
my $OUT;
open($OUT,'>',$fn);
say $OUT "set template=$_";
	$SET_IP_HOME=$HS_iphome   if m/HS/; 
$SET_IP_HOME=$em_iphome   if m/em/;
$SET_IP_HOME=$_700_iphome if m/ARC7/;
	$SET_IP_HOME=$_600_iphome if !m/em/ && !m/ARC7/ && !m/HS/;

say $OUT "$SET_IP_HOME";
say $OUT "$ARChitect2_cl";
close($OUT);

}#---END build_bat---


### ---  main() ---
my $perl_p=@ENV{"perl_p"}; 
chdir($perl_p); 
`mkdir -p arch_bat && rm -f arch_bat/*.bat`;
chdir("arch_bat");

open($ALL_BAT_OUT,'>','all_bat.bat');
say $ALL_BAT_OUT '@echo off';
map{
	out_build_bat( $_ );
say "Generating $_\.bat...";
select(undef,undef,undef,0.015555);
#sleep 0.8 ;
#say $ALL_BAT_OUT "echo Start $_\.bat ...";
$_="..\\arch_bat\\".$_;
say $ALL_BAT_OUT 'start "" "'.$_.'.bat"'
}@template;

close($ALL_BAT_OUT);

