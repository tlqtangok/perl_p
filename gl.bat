@echo off 
::git fetch
SET pcnt=0
FOR %%A IN (%*) DO SET /A pcnt+=1

set tag_no=%1

if %pcnt%==0 (
    git log --all -55 --name-status
) else (
    git log %* -55 --name-status
)
@echo on 
