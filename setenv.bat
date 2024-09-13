@echo off
pushd %0\..
set ROOT=%cd%
::echo %ROOT%

set t=%ROOT%\t
set pro=%ROOT%\pro
set perl_p=%ROOT%\perl_p

set PATH=%PATH%;___SEP___;%ROOT%\pro\Perl\bin;%ROOT%\pro\npp;%ROOT%\pro\vim\vim80;%ROOT%\pro\git\cmd;%ROOT%\pro\git\mingw64\bin;%ROOT%\pro\git\usr\bin;%ROOT%\perl_p
popd 

echo setenv ok
@echo on
