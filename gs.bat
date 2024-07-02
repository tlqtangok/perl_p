@echo off 


git status -s 

git status -s |grepw "^.M" > nul
set ret_code=%ERRORLEVEL%

perl -e "print qq(\n);"

if %ret_code% equ 0 (
    git status -s | grepw "^.M" | repl "^.M" "" |tol | repl "^" " git add  "
)

@echo on 
