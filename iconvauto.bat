@echo off
setlocal enabledelayedexpansion

REM iconvauto.bat - Convert text files to UTF-8 with BOM
REM Version: 2.0.0
REM Author: Copilot and User

REM Display current information
for /f "tokens=2 delims==" %%a in ('wmic os get LocalDateTime /value') do set datetime=%%a
set formatted_time=%datetime:~0,4%-%datetime:~4,2%-%datetime:~6,2% %datetime:~8,2%:%datetime:~10,2%:%datetime:~12,2%
echo Current Date and Time: %formatted_time%
echo Current User: %USERNAME%
echo.

REM Check if any arguments were provided
if "%~1"=="" (
    echo Usage: iconvauto.bat file1 [file2 file3 ...] OR iconvauto.bat *.txt
    echo Converts text files to UTF-8 with BOM encoding
    exit /b 1
)

REM Process file arguments with wildcard expansion
set found_files=0
for %%F in (%*) do (
    set /a found_files+=1
    set "file_path=%%F"
    call :process_file "!file_path!"
)

REM Check if any files were found
if !found_files! EQU 0 (
    echo No files found matching the specified pattern
    exit /b 1
)

echo All files processed
exit /b 0

:process_file
set "file_path=%~1"
echo Processing: %file_path%

if not exist "%file_path%" (
    echo Error: File "%file_path%" not found
    exit /b
)

REM Extract the file extension
for %%i in ("%file_path%") do set "file_ext=%%~xi"

REM Skip script files that should not have BOM
echo %file_ext% | findstr /i /b /c:".bat" /c:".cmd" /c:".ps1" /c:".vbs" /c:".reg" > nul
if %ERRORLEVEL% EQU 0 (
    echo SKIP: Script file that should not be converted
    exit /b
)

REM Check if already UTF-8 with BOM using PowerShell
powershell -Command "$bytes = [System.IO.File]::ReadAllBytes('%file_path%'); if ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) { exit 1 } else { exit 0 }"
if %ERRORLEVEL% EQU 1 (
    echo SKIP: Already UTF-8 with BOM
    exit /b
)

REM Get file type
for /f "usebackq tokens=*" %%i in (`file -b "%file_path%"`) do set "file_type=%%i" & goto :continue_process
:continue_process

echo Type: %file_type%

REM Skip ISO-8859 encoded files
echo %file_type% | findstr /i "ISO-8859" > nul
if %ERRORLEVEL% EQU 0 (
    echo SKIP: ISO-8859 format
    exit /b
)

REM Skip binary files
echo %file_type% | findstr /i "executable binary data ELF PE32 COM compiled" > nul
if %ERRORLEVEL% EQU 0 (
    echo SKIP: Binary file
    exit /b
)

REM Skip document files
echo %file_type% | findstr /i "Microsoft Word Excel PowerPoint PDF Document" > nul
if %ERRORLEVEL% EQU 0 (
    echo SKIP: Document file
    exit /b
)

REM Skip binary files by extension
if "%file_ext%" NEQ "" (
    echo %file_ext% | findstr /i /b /c:".exe" /c:".dll" /c:".obj" /c:".bin" /c:".zip" /c:".rar" /c:".7z" /c:".gz" /c:".tar" /c:".bz2" /c:".xz" /c:".pdf" /c:".doc" /c:".xls" /c:".ppt" /c:".jpg" /c:".png" /c:".mp3" /c:".mp4" > nul
    if %ERRORLEVEL% EQU 0 (
        echo SKIP: Binary file extension
        exit /b
    )
)

REM Check if file has textual content
echo %file_type% | findstr /i "text ASCII UTF Unicode" > nul
if %ERRORLEVEL% NEQ 0 (
    echo SKIP: Not a text file
    exit /b
)

REM Convert file to UTF-8 with BOM
echo CONVERTING to UTF-8 with BOM
powershell -Command "$content = Get-Content -Path '%file_path%' -Raw -ErrorAction SilentlyContinue; if ($content -ne $null) { [System.IO.File]::WriteAllText('%file_path%', $content, [System.Text.Encoding]::UTF8); exit 0; } else { exit 1; }"

if %ERRORLEVEL% EQU 0 (
    echo SUCCESS: Converted to UTF-8 with BOM
) else (
    echo FAILED: Conversion failed
)

echo.
exit /b