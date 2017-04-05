
:: 7-zip usage:   
:: 7z x  test_test.7z    -y  -oARC\
:: ls -> ide folder
	set latest_ide=Z:\ide_builds\ide-2015.12_engbuild_009\Windows

:: ls , will get mwdt_2015_12_rel_win.zip
	set latest_zip_of_mwdt=Z:\mwdt_builds\mwdt_2015_12_rel_005\windows\mwdt_2015_12_rel_win.zip

::uncompress here c:\  -> ARC\xx ARC\xx
pushd C:\ 
rmdir /s /q ARC\java ARC\license ARC\nSIM ARC\MetaWare  > nul
del ARC\setenv_mwdt.bat  ARC\EULA.pdf
echo decompress %latest_zip_of_mwdt%, takes 10 minutes...
7z x -y %latest_zip_of_mwdt% -o.\ > nul
rmdir /s /q C:\ARC\MetaWare\ide >nul
copy /Q /E %latest_ide% C:\ARC\MetaWare >nul
cd ARC
call setenv_mwdt.bat 

popd 
