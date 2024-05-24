@echo off 

set argc=0
for %%x in (%*) do (
    set /a argc+=1
)
::echo %argc%

set cur_dir=%1

if %argc% neq 0 (
pushd %cur_dir%
) else (
pushd %CD%
)

@echo on 
