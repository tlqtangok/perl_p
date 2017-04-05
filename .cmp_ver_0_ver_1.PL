#!perl
use feature qw(say); 


$arch_ver_0="_201403_SP1_eng4";
$arch_ver_1="_201403RC3";
#--- compare ---#
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

my $cmd_perl='perl';
my $arg_file='../.perl_DIR0_DIR1_HTML.PL';
map{
$DIR_0='p_'.$_.$arch_ver_0;
$DIR_1='p_'.$_.$arch_ver_1;
#$compare_log=`perl .perl_DIR0_DIR1_HTML.PL $DIR_0 $DIR_1`;
system("$cmd_perl","$arg_file","$DIR_0","$DIR_1");
#say "i am doing perl $DIR_0 $DIR_1 ..."; 
#say $compare_log; 
}@template;



