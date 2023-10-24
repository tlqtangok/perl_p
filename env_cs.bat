@echo off 
::pushd %t% > nul 
::pushd D:\jd\pro\ana\condabin > nul

call D:\jd\pro\vs2019\Enterprise\Common7\Tools\VsMSBuildCmd.bat  && call D:\jd\pro\vs2019\Enterprise\Common7\Tools\vsdevcmd\ext\roslyn.bat

@echo on 
