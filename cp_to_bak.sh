#!/usr/bin/sh

for i in "$@"
do
#	echo $i
cp_dst=bak.$i.`date "+%Y%m%d%H%M"`
cp -r ${i} ${cp_dst}
echo ${cp_dst}


done
