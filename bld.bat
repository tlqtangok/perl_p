@echo off
:: Usage: bld2 [rel|release|debug]

:: Verify VS environment
where cl >nul 2>&1
if %errorlevel% neq 0 (
    call %perl_p%\env_vs.bat

)

@echo off

:: Parse build configuration
set "BUILD_CONFIG=Debug"
set "CONFIG_NAME=debug"

if "%1"=="" (
    set "BUILD_CONFIG=Debug"
    set "CONFIG_NAME=debug"
) else if /i "%1"=="rel" (
    set "BUILD_CONFIG=Release"
    set "CONFIG_NAME=rel"
) else if /i "%1"=="release" (
    set "BUILD_CONFIG=Release"
    set "CONFIG_NAME=release"
) else if /i "%1"=="debug" (
    set "BUILD_CONFIG=Debug"
    set "CONFIG_NAME=debug"
) else (
    echo Error: Invalid argument. Use: bld [rel^|release^|debug]
    exit /b 1
)

:: Auto-detect .sln file with same folder name
for %%F in (*.sln) do (
    set "SLN_FILE=%%F"
    goto :found_sln
)

echo Error: No .sln file found in current directory.
exit /b 1

:found_sln
echo Building %SLN_FILE% [%BUILD_CONFIG%]...

:: Build with minimal output
msbuild "%SLN_FILE%" /p:Configuration=%BUILD_CONFIG% /v:minimal /nologo /clp:ErrorsOnly

if %errorlevel% equ 0 (
    echo [SUCCESS] %SLN_FILE% - %BUILD_CONFIG%
    
    :: Find and print exe full path
    for /r "x64\%BUILD_CONFIG%" %%E in (*.exe) do (
        echo EXE: %%~fE
    )
) else (
    echo [FAILED] %SLN_FILE% - %BUILD_CONFIG%
    exit /b %errorlevel%
)

exit /b 0
