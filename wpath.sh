#!bash


dirname_=`pwd`
if [[ "$1" != "" ]] ; then
dirname_=`readlink -f $1`	
	
		

fi
cygpath -aw ${dirname_}
