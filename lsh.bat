@echo off
set this_dir=%cd%
::echo %this_dir%
set last_few_num=7
if not "%1" == "" (
set last_few_num=%1
)
pushd %this_dir% > nul

dir /b /o-d  /TW |perl -e "@arr=<>; chomp(@arr); @arr_sort = @arr; map{print qq{ $_\n};}@arr_sort[0..$ENV{last_few_num}];" 
::dir /b /o-d  /TW |perl -e "@arr=<>; chomp(@arr); @arr_sort = sort{-M $a <=> -M $b}@arr; map{print qq{ $_\n};}@arr_sort[0..$ENV{last_few_num}];" 

popd

