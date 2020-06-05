#!/usr/bin/sh
set -e

for i in "$@"
do
#	echo $i
FN=$i
FN_FULL_PATH=$(readlink -f $FN)

basename_fn=$(basename $FN_FULL_PATH)
dirname_fn=$(dirname $FN_FULL_PATH)
cd $dirname_fn
cp_dst=bak.${basename_fn}.`date "+%Y%m%d%H%M"`
cp -r ${basename_fn} ${cp_dst}
echo $dirname_fn/${cp_dst}


done
