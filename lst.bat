@echo off 
date /T|repl " .*$" "" > %tmp%\lst.txt
set /p lst=<%tmp%\lst.txt 
::echo %lst%


if "%1" == "" (
dir *.exe *.dll /o-D |grepw %lst%
) else (
dir %1 *.exe *.dll /o-D |grepw %lst%
)
@echo on 
