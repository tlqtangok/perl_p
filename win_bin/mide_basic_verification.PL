#!perl 
use strict; 
use Win32; 

# wait for creating of this c:\ARC\MetaWare\ide\mideworkspace folder
sleep(60); 

my $ide_ws_dirname=q(c:\ARC\MetaWare\ide\mideworkspace); 
chdir ($ide_ws_dirname); 

my $id_log_filename_pass="mide_basic_verification_pass_log.txt"  ;   #C:\ARC\MetaWare\ide\mideworkspace
my $id_log_filename_failed="mide_basic_verification_failed_log.txt" ; 
my $i=0;
my $MAX_minute=10 ; 

my $time_out_flag=1;
for($i=0;$i<$MAX_minute; $i++){
if (	(-e $id_log_filename_pass) || ( -e $id_log_filename_failed )	){
$time_out_flag=0;
last; 
}
sleep(60); 
}


my $id_msg="INITIAL_MSG"; 
if ( $time_out_flag==0 ){
id_msg='run mide_basic_verification.qft time out'; 
Win32::MsgBox($id_msg);
}

if ( -e $id_log_filename_failed ){
$id_msg= 'run mide_basic_verification.qft failed,'."\n".'check %ide_ws% for detail '; 
Win32::MsgBox($id_msg);
}
if ( -e $id_log_filename_pass  ){
$id_msg= 'run mide_basic_verification.qft pass'; 
}
print $id_msg; 



