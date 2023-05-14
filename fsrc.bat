@echo off 
set fn=%tmp%\fn_all.txt
dir /s /b > %fn%
perl %perl_p%\filter_src_fn.PL %fn% 
@echo on

