@echo off 
pushd %0\..
perl %perl_p%\cbin.PL %1 > %perl_p%\cbin_log.bat
::type %perl_p%\cbin_log.bat
popd 
call %perl_p%\cbin_log.bat 

@echo on 