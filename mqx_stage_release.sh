

#!/bin/bash

#
# cygwin script used to stage an MQX release 

# --- GLOBAL VAR ---
mqx_root_old=$1 # mqx_cp_from_path
REL=$2  		# mqx2.61
# --- END GLOBAL VAR ---
#DEST=/cygdrive/c/ARC
DEST=`dirname $mqx_root_old`

if [ "$mqx_root_old" == "" ] || [ "$REL" == "" ]; then
	#	export CP_FROM=$DEST/software/mqx2.61_old/MQX_2.61   # in the MQX_ROOT , dir of build 
	echo -e " Usage:\n\tsh mqx_stage_release.sh /cygdrive/c/arc/software/MQX_2.61  mqx2.61 "
	exit 1;
fi

export CP_FROM=$mqx_root_old

export DOC_PATH=$3

#TODO: create software if not there.
# for now, just mkdir this outside before running this script
echo -n "pushd " && pushd $DEST

echo "rm old $REL... " && rm -rf $REL 
mkdir $REL
echo "copy from $CP_FROM to `pwd`/$REL, 1 min needs... "
cp -R $CP_FROM/.  $REL
#cp -r $CP_FROM/.* $REL


#
# Remove stuff NOT delivered to customers
#
echo -n "pushd " && pushd $REL
echo rm -rf CodeSizeReports
rm -rf CodeSizeReports
echo rm -rf ktest
rm -rf ktest
echo rm -rf not_released
rm -rf not_released
echo rm -rf TimingReports
rm -rf TimingReports
echo rm -rf mqxenv.bat
rm -rf mqxenv.bat


#----------------------------------
echo Change permissions ...
#
alias find_by_name='find ./ -name '
find_by_name '*' | perl -e ' @arr=<>; map{chomp; q(").$_.q(");}@arr;@_0777=grep /(\.exe$)|(\.bat$)|(\.sh$)/,@arr;	chmod 0777,@_0777; @_0666=grep /(\.h$)|(\.mk$)|(\.met$)|(\.c$)|(\.S$)|(\.s$)|(Makefile$)|(\.makefile$)/,@arr;	chmod 0666,@_0666;	@_0755=grep /(\.cproject$)|(\.project$)|(\.pref$)|(\.defs$)|(\.lnk$)|(\.xml$)/,  @arr; chmod 0755,@_0755 ;'


#
# DOCS
#
if ! [ "DOC_PATH" != "" ]; then
mkdir -p docs
cp -rp $DOC_PATH/. docs
#cp -rp $DOC_PATH/.* docs
fi

echo -n "popd " && popd
echo -n "popd " && popd

#
# done
#
echo "done."

