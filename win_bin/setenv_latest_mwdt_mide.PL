#!perl 
use strict; 
my $cmd=q(
:: ### windows batch file ### ::

:: 7-zip usage:  7z x  test_test.7z    -y  -oARC\
:: ls -> ide folder
	set latest_ide=Z:\ide_builds\LATEST_MIDE\Windows

:: ls , will get mwdt_2015_12_rel_win.zip
	set latest_zip_of_mwdt=LATEST_ZIP_OF_MWDT

::uncompress here c:\  -> ARC\xx ARC\xx
pushd C:\ 
rmdir /s /q ARC\java ARC\license ARC\nSIM ARC\MetaWare  > nul
del ARC\setenv_mwdt.bat  ARC\EULA.pdf
echo. - decompress %latest_zip_of_mwdt%, takes 10 minutes...
7z x -y %latest_zip_of_mwdt%  > nul
rmdir /s /q C:\ARC\MetaWare\ide >nul
xcopy /Q /E %latest_ide% C:\ARC\MetaWare >nul
cd ARC
call setenv_mwdt.bat 

:: run test cases ::
perl %perl_p%\cc_test.PL
:: end testcases 

copy %perl_p%\latest_info.log %perl_p%\latest_info.log.bak
popd 




:: ### windows batch file end ### ::
); 


my $perl_p=@ENV{'perl_p'}; 
my $perl_p_file=$perl_p.'\latest_info.log'; 
my  ($latest_mwdt, $latest_mide,$flag_up_to_date)=&get_latest_mwdt_latest_mide($perl_p_file); 

my $latest_zip_of_mwdt = &full_name_of_mwdt_zip($latest_mwdt); 	#Z:\mwdt_builds\mwdt_2015_12_rel_005\windows\mwdt_2015_12_rel_win.zip
print "- latest version -\n"; 
print "- MWDT:\t$latest_mwdt", "\n", "- MIDE:\t$latest_mide", "\n\n"; 

my %hash=("LATEST_MIDE",$latest_mide, "LATEST_ZIP_OF_MWDT",$latest_zip_of_mwdt); 

$cmd=&subst_by_hash($cmd, \%hash ); 
open my $FP, ">", 'c:\ARC\run_latest_mwdt_mide.bat'; 
print $FP $cmd;
if ($flag_up_to_date==0){
print '- type	C:\ARC\run_latest_mwdt_mide.bat'."\n- run\n\tC:\\ARC\\run_latest_mwdt_mide.bat"; 
print "- your mwdt+mide is not the latest version!\n"; 
}
else{
print "\n- mwdt+mide is up to date\n"; 
}
close $FP; 




### sub list ###
sub get_latest_mwdt_latest_mide($){
my $perl_p_file=@_[0]; 

if (! -e $perl_p_file ){die "- no exist $perl_p_file\nplease run >latest_info on arcdev4\n"; }; 

my @latest_info_content= `type $perl_p_file`; 
my $perl_p_file_bak="$perl_p_file.bak"; 
my @former_latest_info_content=`type $perl_p_file_bak`; 
my $flag_up_to_date=0; 

if ("@former_latest_info_content" eq "@latest_info_content" ){
#print "- mwdt and mide is up to date\n"; 
$flag_up_to_date=1; 
}
else {
#print "- your mwdt+mide is not the latest version!\n"; 
}


my @latest_mwdt_=grep m/^mwdt/, @latest_info_content; 
my @latest_mide_=grep m/^ide/, @latest_info_content; 
chomp(@latest_mwdt_[0]); chomp(@latest_mide_[0]); 
my $latest_mwdt=@latest_mwdt_[0]; 
my $latest_mide=@latest_mide_[0]; 
return ($latest_mwdt, $latest_mide,$flag_up_to_date); 
}
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


