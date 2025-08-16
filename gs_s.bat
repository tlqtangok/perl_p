@echo off 


git submodule foreach "git status -s"|perl -pe  "chomp; if(m/^M.? |^ M.? |Entering /){$_=$_.qq(\n);;}else{$_=qq();}" | tee %tmp%\gs_s.log

type %tmp%\gs_s.log |grepw "^.M" > nul

set ret_code=%ERRORLEVEL%

perl -e "print qq(\n);"

if %ret_code% equ 0 (
perl -e "print qq{git add };"
    type %tmp%\gs_s.log | perl -pe  "chomp; if(m/^M.? |^ M.? /){   s/M.? //; $_=$_.qq( );;}else{$_=qq();}" 
perl -e "print qq{\n};"
)

@echo on 
