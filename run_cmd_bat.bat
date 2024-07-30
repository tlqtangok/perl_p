@echo off
pushd %0\..
where cl > nul 
if %errorlevel% NEQ 0 (echo error , no cl && exit /b 1)

perl run.PL %1

popd
exit /b 0

@echo off

