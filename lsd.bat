@echo off 

for %%i in (A B C D E F G H I J K L M N O P Q R S T U V W X Y Z) do (if exist %%i:\ echo %%i:)



perl -e " my @fc = `net use|grepw OK`;   if (@fc){$_=$fc[0]; my @a=split m/\s+/; print qq(\n),$a[1] , qq( => ), $a[2],qq(\n);  }"; 
@echo on 
