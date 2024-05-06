@echo off 

pushd %code% > nul 

if "%1"=="uarm" (
pushd ..\..\scan_uarm\scan > nul
)

@echo on 
