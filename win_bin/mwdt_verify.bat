@echo off

echo -

set |grep_ METAWARE_ROOT
set |grep_ NSIM_HOME
where ccac 
::ccac -version 
where mide 

echo -

echo - verify VM
%METAWARE_ROOT%\..\jre\bin\java -version |grep_ VM

echo -

echo - verify shortcuts
pushd %startm%\MWDT*
tree  /F
cd Programs
ecd 

echo -

@echo on
popd 
