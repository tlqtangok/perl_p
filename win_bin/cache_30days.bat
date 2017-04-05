@echo off 
set LICENSE_30_DAY_CACHE=1
pushd %0\..
echo ^#include ^<stdio.h^> > test.c && echo int main(){return 0;}; >> test.c

echo. - build em
ccac -av2em test.c 
echo. - dbg em
mdb -av2em -cl a.out 




:: check cache, should see 30 days.
set LICENSE_30_DAY_CACHE=1
echo unset SNPSLMD_LICENSE_FILE
set SNPSLMD_LICENSE_FILE=

echo. - build em
ccac -av2em test.c 
echo. - dbg em
mdb -av2em -cl a.out 



popd 
@echo on 