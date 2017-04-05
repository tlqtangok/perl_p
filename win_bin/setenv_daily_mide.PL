#!perl
my $batch_file = 'c:\arc\run_setenv_daily_mide.bat';
my $eclipse_dir_of_mwdt = 'C:\ARC\MetaWare\ide\eclipse'; 
my $_7z_exe = 'C:\Program Files\7-Zip\7z.exe' ; 
my $perl_p = 'P:\tmp\perl_p' ; 
my $cmd_test = "call $perl_p\\lsh.bat". ' z:\mide\IDE_BUILD_RELEASES' ; 
my @arr_check_in_id =`$cmd_test`; 
chomp(@arr_check_in_id[0]); @arr_check_in_id[0] =~ s/\s//g;
my $LATEST_CHECK_IN_ID = @arr_check_in_id[0]; 
print "- latest daily mide is : $LATEST_CHECK_IN_ID\n"; 

#z:\mide\IDE_BUILD_RELEASES\723758_thuymain\products\com.arc.ide-win32.win32.x86.zip
my $cmd_repl = q(
REM set env for daily mide 
REM 7z x -y com.arc.ide-win32.win32.x86.zip -o%cd%\test >nul
rmdir /s /q $eclipse_dir_of_mwdt
mkdir $eclipse_dir_of_mwdt >nul
"$_7z_exe" x -y z:\mide\IDE_BUILD_RELEASES\$LATEST_CHECK_IN_ID\products\com.arc.ide-win32.win32.x86.zip -o$eclipse_dir_of_mwdt >nul
);
$cmd_repl =~ s/\$perl_p/$perl_p/; 
$cmd_repl =~ s/\$_7z_exe/$_7z_exe/; 
$cmd_repl =~ s/\$LATEST_CHECK_IN_ID/$LATEST_CHECK_IN_ID/; 
$cmd_repl =~ s/\$eclipse_dir_of_mwdt/$eclipse_dir_of_mwdt/mg;

open my $FP_0,'<', $batch_file ; 
my @arr_file_content = <$FP_0>; 
close $FP_0; 

# == 0 means no log the latest 
@arr_file_content = grep m/$LATEST_CHECK_IN_ID/, @arr_file_content ; 





if ( @arr_file_content == 0 ) {
open my $FP, '>', $batch_file; 
print $FP $cmd_repl ; 
close $FP; 

print "- run \n".$batch_file."\n"; 
system( $batch_file ); 



}
else {
print "- mide is already latest daily mide!\n"; 
}


