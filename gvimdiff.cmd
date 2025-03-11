:: git difftool f2e61689e2582a3864c88f027c4e0d46906954bf 37465d22b558bee8a504be9dbbb9173ad491438b
::
:: git config --global difftool.prompt false  
:: git difftool HEAD origin/master
:: git config --global diff.tool gvimdiff
:: git config --global difftool.gvimdiff.cmd "gvimdiff.cmd $LOCAL $REMOTE"
:: git config --global difftool.gvimdiff.trustExitCode true
:: git config --global difftool.prompt false
::
::
:: git config --global merge.tool gvim
:: git config --global mergetool.gvim.cmd "gvimdiff.cmd $LOCAL $REMOTE $MERGED"
:: git config --global mergetool.gvim.trustExitCode true
:: git merge --abort
:: git push
:: git pull 


@echo off

:: pcnt is input params number
SET pcnt=0  
FOR %%A IN (%*) DO SET /A pcnt+=1  

:: Exit with code -1 if parameter count is not 2 or 3
if not %pcnt% EQU 2 (
  if not %pcnt% EQU 3 (
    echo Error: Script requires exactly 2 or 3 parameters.
    echo Usage:
    echo   %~nx0 LOCAL_FILE REMOTE_FILE              - Compare two files
    echo   %~nx0 LOCAL_FILE REMOTE_FILE RESULT_FILE  - Compare and save to result
    exit /b -1
  )
)

if %pcnt% EQU 2 (
  setlocal  

  echo LOCAL: %~1  
  echo REMOTE: %~2  

  REM Start gvimdiff  
  "gvim.exe" -f -d "%~1" "%~2"  

  endlocal  
  exit /b %errorlevel%  
)

if %pcnt% EQU 3 (
  setlocal  

  echo LOCAL: %~1  
  echo REMOTE: %~2  
  echo TO_SAVE: %~3

  echo cp -p  "%~2" "%~3"  
  cp -p  "%~2" "%~3"  2>nul

  if %errorlevel% neq 0 (
    echo Error: Failed to copy "%~2" to "%~3"
    exit /b 1
  )
  
  REM Start gvimdiff  
  "gvim.exe" -f -d "%~1" "%~3"  
  
  endlocal
  exit /b %errorlevel%
)
