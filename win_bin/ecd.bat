@echo off
if  "%1%"=="" (
::echo %1%
pushd %cd%
explorer.exe /e, %cd%
popd
)else explorer.exe /e, %1%

@echo on 