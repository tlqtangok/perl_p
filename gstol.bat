@echo off 
git status -s 
perl -e "print qq(\n);"

git status -s | grepw ".M" | repl "^.M" "" |tol | repl "^" " git add  "
@echo on 
