@echo off

dir /b %1 %2 %3 %4 %5 %6 > %tmp%\full_.txt 
perl %perl_p%\full_.PL %CD% %tmp%\full_.txt | repl "\\" "/"
@echo on
