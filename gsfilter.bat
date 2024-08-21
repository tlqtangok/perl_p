@echo off 

call C:\jd\perl_p\gs.bat |grepw /V /C:"git add"

@echo off 

echo ============

perl -e "print qq{\n};"
::C:\jd\perl_p\gs.bat | grepw /C:"git add" | perl -pe "print qq{git add "};@a=split m/ /; @a_=();map{ ;}@a;" 
C:\jd\perl_p\gs.bat | grepw /C:"git add" | perl -pe ";@a=split m/ /;$a_=qq{git add }; map{ if (-f $_) {  if (m/\.lib$|tags$/){} else{$a_.=qq{$_ }; } } }@a; $_= $a_;" 

perl -e "print qq{\n};"
@echo on 
