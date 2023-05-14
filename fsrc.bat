@echo off 
:: can work on linux
set fn=%tmp%\fn_all.txt
dir /s /b > %fn% 
dos2unix %fn% 2>nul 
perl %perl_p%\filter_src_fn.PL %fn%  %1 %2 %3 %4 %5 %6 %7 %8
@echo on
