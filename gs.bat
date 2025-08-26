@echo off 

set submodules_tol=NULL_SUBMODULES
if  exist .gitmodules (
cat .gitmodules|grep path |perl -pe "s/^.*path = //;"|tol > %tmp%\submodules.txt
set /p submodules_tol=<%tmp%\submodules.txt
) 

::echo %submodules_tol%

::cat .gitmodules|grep path |perl -pe "s/^.*path = //;"|tol > %tmp%\submodules.txt
::set /p submodules_tol=<%tmp%\submodules.txt
git status -s |grepw /V "%submodules_tol%" |perl -pe  "chomp; if(m/^M.? |^ M.? /){$_=$_.qq(\n);;}else{$_=qq();}"

git status -s |grepw /V "%submodules_tol%"  |grepw "^.M" > nul
set ret_code=%ERRORLEVEL%

perl -e "print qq(\n);"

if %ret_code% equ 0 (
perl -e "print qq{git add };"
    git status -s |grepw /V "%submodules_tol%" | perl -pe  "chomp; if(m/^M.? |^ M.? /){   s/M.? //; $_=$_.qq( );}else{$_=qq();}" 
perl -e "print qq{\n};"
)

@echo on 
