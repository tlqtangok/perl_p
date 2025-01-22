@echo off
set E_OPTS=%0
echo %E_OPTS%

::exit /b 0
perl %perl_p%\git_opts.PL %E_OPTS%
@echo on 
