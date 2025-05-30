@echo off
setlocal enabledelayedexpansion

set service_name=sshd_test
set display_name=OpenSSH Server Test
set description=OpenSSH SSH daemon (Test Service)
set "service_exe=C:\Program Files\OpenSSH\sshd.exe"
set "service_dir=C:\Program Files\OpenSSH"
set "service_log_dir=%userprofile%\OpenSSH\logs"
set "log_stdout=sshd_stdout.log"
set "log_stderr=sshd_stderr.log"
set service_startup=SERVICE_AUTO_START
set service_account=LocalSystem

echo ===================================================
echo Creating OpenSSH service with NSSM as "%service_name%"
echo ===================================================
echo.

echo [COMMAND] net session ^>nul 2^>^&1
REM Check for Administrator privileges
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: This script requires Administrator privileges.
    echo Please right-click on the script and select "Run as administrator".
    pause
    exit /b 1
)
echo [SUCCESS] Script is running with Administrator privileges.

echo [COMMAND] where nssm ^>nul 2^>^&1
REM Check if NSSM is available in PATH or in the current directory
where nssm >nul 2>&1
if %errorlevel% neq 0 (
    echo [COMMAND] if exist .\nssm.exe...
    if exist .\nssm.exe (
        echo [INFO] Using local NSSM executable
        set "NSSM_CMD=.\nssm.exe"
        echo [SET] NSSM_CMD=.\nssm.exe
    ) else (
        echo ERROR: NSSM is not found in PATH or in the current directory.
        echo Please download NSSM from https://nssm.cc/download and place nssm.exe in this directory,
        echo or include its directory in your PATH environment variable.
        pause
        exit /b 1
    )
) else (
    set "NSSM_CMD=nssm"
    echo [SET] NSSM_CMD=nssm
    echo [SUCCESS] Found NSSM in PATH.
)

echo [COMMAND] if not exist "%service_exe%" ...
REM Check if sshd.exe exists
if not exist "%service_exe%" (
    echo ERROR: sshd.exe not found at %service_exe%.
    echo Please make sure OpenSSH is installed correctly.
    pause
    exit /b 1
)
echo [SUCCESS] Found sshd.exe at %service_exe%.

echo [COMMAND] if not exist "%service_log_dir%" ...
REM Create logs directory if it doesn't exist
if not exist "%service_log_dir%" (
    echo [INFO] Creating logs directory...
    echo [COMMAND] mkdir "%service_log_dir%"
    mkdir "%service_log_dir%"
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create logs directory.
        pause
        exit /b 1
    )
    echo [SUCCESS] Created logs directory at %service_log_dir%.
) else (
    echo [INFO] Logs directory already exists at %service_log_dir%.
)

echo [COMMAND] %NSSM_CMD% status %service_name% ^>nul 2^>^&1
REM Remove existing service if it exists
echo [INFO] Checking for existing service...
%NSSM_CMD% status %service_name% >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] Found existing service. Removing...
    echo [COMMAND] %NSSM_CMD% remove %service_name% confirm
    %NSSM_CMD% remove %service_name% confirm
    if %errorlevel% neq 0 (
        echo ERROR: Failed to remove existing service.
        pause
        exit /b 1
    )
    echo [SUCCESS] Existing service removed successfully.
    echo [COMMAND] timeout /t 2 ^>nul
    timeout /t 2 >nul
) else (
    echo [INFO] No existing service found.
)

REM Install the service
echo [INFO] Installing %service_name% service...
echo [COMMAND] %NSSM_CMD% install %service_name% "%service_exe%"
%NSSM_CMD% install %service_name% "%service_exe%"
if %errorlevel% neq 0 (
    echo ERROR: Failed to install service.
    pause
    exit /b 1
)
echo [SUCCESS] Service installed.

REM Configure service parameters
echo [INFO] Configuring service parameters...

echo [COMMAND] %NSSM_CMD% set %service_name% DisplayName "%display_name%"
%NSSM_CMD% set %service_name% DisplayName "%display_name%"
echo [COMMAND] %NSSM_CMD% set %service_name% Description "%description%"
%NSSM_CMD% set %service_name% Description "%description%"
echo [COMMAND] %NSSM_CMD% set %service_name% AppDirectory "%service_dir%"
%NSSM_CMD% set %service_name% AppDirectory "%service_dir%"
echo [COMMAND] %NSSM_CMD% set %service_name% AppStdout "%service_log_dir%\%log_stdout%"
%NSSM_CMD% set %service_name% AppStdout "%service_log_dir%\%log_stdout%"
echo [COMMAND] %NSSM_CMD% set %service_name% AppStderr "%service_log_dir%\%log_stderr%"
%NSSM_CMD% set %service_name% AppStderr "%service_log_dir%\%log_stderr%"
echo [COMMAND] %NSSM_CMD% set %service_name% Start %service_startup%
%NSSM_CMD% set %service_name% Start %service_startup%
echo [COMMAND] %NSSM_CMD% set %service_name% ObjectName %service_account%
%NSSM_CMD% set %service_name% ObjectName %service_account%

echo.
echo [SUCCESS] Service configuration complete.

REM Start the service
echo [INFO] Starting %service_name% service...
echo [COMMAND] %NSSM_CMD% start %service_name%
%NSSM_CMD% start %service_name%
if %errorlevel% neq 0 (
    echo [WARNING] Failed to start service. Check logs for details.
    echo [INFO] You can check the log files or Windows Event Viewer for more information.
) else (
    echo [SUCCESS] Service started successfully.
)

REM Display service status
echo.
echo [INFO] Current service status:
echo [COMMAND] %NSSM_CMD% status %service_name%
%NSSM_CMD% status %service_name%

echo.
echo ===================================================
echo [SUCCESS] Installation complete!
echo.
echo Service Details:
echo   - Name: %service_name%
echo   - Display Name: %display_name%
echo   - Executable: %service_exe%
echo   - Working Directory: %service_dir%
echo   - Log Files:
echo     * %service_log_dir%\%log_stdout%
echo     * %service_log_dir%\%log_stderr%
echo.
echo Use the following commands to manage the service:
echo   - %NSSM_CMD% status %service_name%   : Check service status
echo   - %NSSM_CMD% stop %service_name%     : Stop the service
echo   - %NSSM_CMD% start %service_name%    : Start the service
echo   - %NSSM_CMD% restart %service_name%  : Restart the service
echo   - %NSSM_CMD% edit %service_name%     : Edit service configuration
echo   - %NSSM_CMD% remove %service_name% confirm : Remove the service
echo ===================================================

echo [INFO] Press any key to exit...
pause
endlocal