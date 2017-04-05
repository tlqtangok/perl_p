@echo off

set JAVA_HOME=%pro%\jd_java

pushd %cd%

set Path=%JAVA_HOME%\bin;%JAVA_HOME%\jre\bin;%Path%

::..\eclipse\eclipse.exe


::java 
popd 


@echo on 