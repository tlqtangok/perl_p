
set git_cmd=git_gvim_diff_merge.bat

git config --global diff.tool gvimdiff
git config --global difftool.gvimdiff.cmd "%git_cmd% $LOCAL $REMOTE"
git config --global difftool.gvimdiff.trustExitCode true
git config --global difftool.prompt false
::
::
git config --global merge.tool gvim
git config --global mergetool.gvim.cmd "%git_cmd% $LOCAL $REMOTE $MERGED"
git config --global mergetool.gvim.trustExitCode true

:: >> if ok
:: git add xxxx.txt 
:: git commit -m "xxx"
:: git push 
::
:: >> if not need 
:: git merge --abort
:: git pull 
:: git mergetool 
::
:: >> if push error,the git stash clear will invalid, need run:
:: gl
:: git reset HEAD~1
:: git_save
:: git_reset
:: git_merge
::

