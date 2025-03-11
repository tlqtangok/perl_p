@echo off 


SET pcnt=0  
FOR %%A IN (%*) DO SET /A pcnt+=1  

perl %perl_p%\git_gvim_diff_merge.PL "%~1" "%~2" "%~3"

@echo on
