@echo off

:: reg add "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Command Processor" /v AutoRun /t REG_SZ /d "d:\jd\perl_p\alias_win.cmd" /f
::
::doskey tarc=tar czf $*
doskey tarc=tar cvzf $* ^>%TEMP%\tarc.log 2^>^&1 ^&^& ^( ^( findstr /N "^" %TEMP%\tarc.log ^| findstr /B /L "11:" ^>NUL 2^>^&1 ^&^& ^( head -n 5 %TEMP%\tarc.log ^&^& echo ... ^&^& tail -n 5 %TEMP%\tarc.log ^) ^|^| ^( type %TEMP%\tarc.log ^) ^) ^| uniq ^)
doskey tart=tar tvzf $*
::doskey tarx=tar xvzf $* > %tmp%\tarx.log 2>&1 &&  (head -n 5 %tmp%\tarx.log && echo ... && tail -n 5 %tmp%\tarx.log ) | uniq
::
::doskey tarx=tar xvzf $* ^> %tmp%\tarx.log 2^>^&1 ^&^& ^(head -n 5 %tmp%\tarx.log ^&^& echo ... ^&^& tail -n 5 %tmp%\tarx.log ^) ^| uniq
::
doskey tarx=tar xvzf $* ^>%TEMP%\tarx.log 2^>^&1 ^&^& ^( ^( findstr /N "^" %TEMP%\tarx.log ^| findstr /B /L "11:" ^>NUL 2^>^&1 ^&^& ^( head -n 5 %TEMP%\tarx.log ^&^& echo ... ^&^& tail -n 5 %TEMP%\tarx.log ^) ^|^| ^( type %TEMP%\tarx.log ^) ^) ^| uniq ^)
::call d:\jd\setenv.bat


