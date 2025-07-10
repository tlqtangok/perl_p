@echo off
setlocal
set TAR_LOG=%TEMP%\tarc.log
tar cvzf %* > "%TAR_LOG%" 2>&1
findstr /N "^" "%TAR_LOG%" | findstr /B /L "11:" >NUL 2>&1
if %errorlevel%==0 (
    head -n 5 "%TAR_LOG%"
    echo ...
    tail -n 5 "%TAR_LOG%"
) else (
    type "%TAR_LOG%"
) | uniq
endlocal
@echo on
