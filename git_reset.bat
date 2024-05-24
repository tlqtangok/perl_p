@echo off 
::- git overwrite local files 
git fetch --all  
git reset --hard origin/master
git pull
@echo on 
