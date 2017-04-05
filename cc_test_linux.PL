#!perl 
use strict; 
my $cmd=q(
### windows batch file ### ::

which ccac  
#echo
which mide 

# run test cases ::
echo - run test cases for cc...
#test_c=$tmp/test.c
echo \#include \<stdio.h\> >test.c
echo int main\\\(\\\)\{return 0\;\}\; >>test.c

#run_testcase_log=$tmp/run_testcase.log

  mcc -a6 test.c > run_testcase.log 2>&1
  mdb -a6 -cl -run a.out >> run_testcase.log 2>&1
  mcc -a7 test.c >> run_testcase.log 2>&1
#  mdb -a7 -cl -run a.out >> run_testcase.log 2>&1 
  ccac -av2em test.c >> run_testcase.log 2>&1
  mdb  -av2em -cl -run a.out >> run_testcase.log 2>&1
  ccac -av2hs test.c >> run_testcase.log 2>&1
#  mdb  -av2hs -cl -run a.out >> run_testcase.log 2>&1
# @echo off


#@echo on 
# ### windows batch file end ### ::
); 

### initial() ###
# I must chdir from c:\windows\system32, because of mcc bug
#################
my $pro_dir = @ENV{tmp}; 
#my $pro_dir=`echo $tmp`; 
chomp($pro_dir); 

#print $pro_dir."-------------------------\n"; 

chdir ( $pro_dir ); 
### end initial() ###

open my $FP, ">","test_cc.sh"; 
print $FP $cmd; 
#print $cmd; 

close $FP; 

system( q(sh test_cc.sh ) ); 
my @run_case_log_content=`cat run_testcase.log`; 


my $error_status=0; 
$error_status++ if "@run_case_log_content" =~ m/invalid\:/mi ; 
$error_status++ if "@run_case_log_content" =~ m/error/mi ; 
$error_status++ if "@run_case_log_content" =~ m/unknown/mi ; 
$error_status++ if "@run_case_log_content" =~ m/abort/mi; 
$error_status++ if "@run_case_log_content" =~ m/11\:SIGNAL\[11\]/mi; 
$error_status++ if "@run_case_log_content" =~ m/Front-end called failed/mi; 

if ($error_status!=0){
print "\n- verify Failed, exit code $error_status\n"; 
print "\tplease see \$tmp/run_testcase.log for details...\n"; }
else {
print "\n- verify cc success\n"; 
if( @ARGV == 0){
#system( 'del run_testcase.log ' ); 
}
}

system('del a.out test.c  test_cc.sh'); 







### sub list ###
sub get_latest($){
my $latest_cmd=@_[0]; 
my @latest_array=(); 

#@latest_array=`pushd Z:\ide_builds && lsh `;
@latest_array=`$latest_cmd `;

my $i=0;
for ($i=0;$i<@latest_array; $i++){
my $tmp=@latest_array[$i]; 
chomp($tmp); $tmp=~s/\s//g; 
if ( $tmp eq 'latest'){
last;
}
} 
my $latest_mide=@latest_array[$i+1]; 
chomp($latest_mide); 
$latest_mide =~ s/\s//g; 
#print "-------".$latest_mide."---------------"; 
return $latest_mide; 


}
sub full_name_of_mwdt_zip($){
my $latest_mwdt_ =@_[0]; 
my $cmd_zip=q( pushd Z:\mwdt_builds\LATEST_MWDT\windows\ && dir /s /b *.zip ); 
$cmd_zip =~ s/LATEST_MWDT/$latest_mwdt_/; 
# run get zip 
$cmd_zip=`$cmd_zip`; chomp($cmd_zip); 
return $cmd_zip ; 


}
sub subst_by_hash($src \%hash_){
my $str=@_[0]; 
my $hash_=@_[1]; 
my %hash=%$hash_; 
#print %hash; 
my @key_arr=keys %hash; 
map{
$str =~ s/$_/@hash{$_}/m; 
}@key_arr; 

#print "\n", $str; 

return $str; 


} # end sub substi_xx()
