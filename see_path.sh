#!bash
echo $PATH|perl -e '$_=<>; @arr=split m/\:/;map{print $_."\n";}@arr;'
