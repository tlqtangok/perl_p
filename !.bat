@echo off 


SET pcnt=0  
FOR %%A IN (%*) DO SET /A pcnt+=1  



set MYLINENO=%1
if %pcnt% EQU 1 (
    perl %perl_p%\get_e_cml_from_his.PL  %MYLINENO%
    set e_his_cmd=%tmp%\e_his_cmd.bat;
    call %e_his_cmd%
)

@echo on
