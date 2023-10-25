@echo off 

set cur_dir=%1

if not "%cur_dir%"=="" (
pushd %1
) else (
pushd %CD%
)

@echo on 
