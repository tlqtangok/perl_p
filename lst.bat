@echo off 
::date /T|repl " .*$" "" > %tmp%\lst.txt

date /T |perl -pe "s/^.*(20\d\d.\d\d.\d\d).*$/\1/; " > %tmp%\lst.txt
set /p lst=<%tmp%\lst.txt 
echo %lst%

set filter_args=*.exe *.dll


if "%1" == "" (
dir /o-d %filter_args% |grepw %lst%
) else (
dir /o-d %1 %filter_args% |grepw %lst%
)
@echo on 
