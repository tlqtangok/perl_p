@echo off


set argc=0
for %%x in (%*) do (
    set /a argc+=1
)
::echo %argc%

if  %argc% equ 0 (
::echo %1%
pushd %cd%
explorer.exe /e, %cd%
popd
)else explorer.exe /e, %1%

@echo on 
