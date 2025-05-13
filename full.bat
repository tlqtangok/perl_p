@echo off
setlocal enabledelayedexpansion

::echo File Path Listing - %date% %time%
::echo --------------------------------------

if "%~1"=="" (
    echo Error: No parameters provided
    echo Usage: %~nx0 [file or folder] [file or folder] ...
    goto :end
)

:process
if "%~1"=="" goto :end

set "item=%~1"
set "item=%item:/=\%"

if exist "%item%" (
    for %%i in ("%item%") do echo  %%~fi
) else (
    echo   [ERROR] Path not found: %item% 1>&2
)


shift
goto :process





:end
::echo --------------------------------------
::echo Processing complete
::

::echo.

@echo on 
