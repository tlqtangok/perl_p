@echo off 


git status -s|perl -pe  "chomp; if(m/^M.? |^ M.? /){$_=$_.qq(\n);;}else{$_=qq();}"

git status -s |grepw "^.M" > nul
set ret_code=%ERRORLEVEL%

perl -e "print qq(\n);"

if %ret_code% equ 0 (
perl -e "print qq{git add };"
    git status -s | perl -pe  "chomp; if(m/^M.? |^ M.? /){   s/M.? //; $_=$_.qq( );;}else{$_=qq();}" 
perl -e "print qq{\n};"
)

@echo on 
