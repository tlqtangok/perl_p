@echo off
set this_dir=%cd%
::echo %this_dir%
if not "%1" == "" (
set this_dir=%1
)
pushd %this_dir% > nul
dir /b /o-D  |perl -e "@arr=<>; map{print ' '.@arr[$_];}(0..7);"
popd
