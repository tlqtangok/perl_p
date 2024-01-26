@echo off 
pushd %0\..\ >nul 2>&1
set ROOT=%CD%
:: jd env list
set t=%ROOT%\t
set pro=%ROOT%\pro
set dl=%ROOT%\dl
set perl_p=%ROOT%\perl_p


set PATH=%PATH%;%perl_p%;%pro%\Perl\bin

@echo on 


