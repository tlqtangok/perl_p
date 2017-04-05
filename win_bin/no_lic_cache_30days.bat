@each off 
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
