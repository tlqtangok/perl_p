@echo off
set E_OPTS=%0   
:: is git_save
::echo %E_OPTS%

::exit /b 0
perl %perl_p%\git_opts.PL %E_OPTS%
@echo on 
