@echo off 
::pushd \\192.168.16.206\软件安装包\预装软件

lsd |grepw /I Z  > nul

if %errorlevel% equ 0 (
pushd z:\
goto EOF_
)

set p206=\\192.168.16.206\smb_share_2T
pushd %p206% > nul 

:EOF_

@echo on 
