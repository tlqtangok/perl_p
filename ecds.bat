@echo off
if  "%1%"=="" (
::echo %1%
pushd %cd%
explorer.exe /e, %cd%
popd
)else start explorer.exe /select,%1%

@echo on 
