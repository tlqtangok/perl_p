#!bash 

all_bat_name=all_makefile_run.bat # run_bat *.bat , run_bat is a alias
build_mqx_log=a6_a7_em_hs_build.log

wtee_to_log="  wtee -a $build_mqx_log "
function Usage(){
echo " Usage: "
echo "	cd  MQX_ROOT/mqx2.60/build "
echo "	sh mqx_run_all_makefile_in_bat.sh [-s|-b|-c]	"
echo "	-s) == save the all_makefile_run.bat "
echo "	-b) == make -f em_makefile ,build all"
echo "  -c) == make -f em_makefile clean"
exit 1
}

# save the FLAGs, is num, so use "" -eq ""
build_makefile_flag=0 
save_flag=0
clean_flag=0     #default, not clean the mqx.a

while getopts 'sbc?' argv ; do  #s,save bat; b, build makefile; c, make clean ;
	case $argv in
		s) 
			save_flag=1;;
		b)      
			build_makefile_flag=1;;
		c)	clean_flag=1;;
	       \?) 
		Usage ;; 
esac
done
alias perl_mqx_config='yes| perl $perl_p/mqx_makefile_auto_config.PL '

# get all a6 a7 em hs_makefile 
perl_mqx_config a6 && cp makefile a6_makefile 
perl_mqx_config a7 && cp makefile a7_makefile 
perl_mqx_config em && cp makefile em_makefile 
perl_mqx_config hs && cp makefile hs_makefile 


var_to_bat=""
var_to_build_log=""
echo $var_to_bat > $all_bat_name  # clear the *bat file
echo $var_to_build_log > $build_mqx_log   # clear build_log file

mqx_p=`cygpath -aw \`pwd\` ` # get mqx2.60/build path in dos format 
echo pushd $mqx_p >> $all_bat_name

my_makefiles=`ls *_makefile 2>/dev/null `

# do make -f a6 a7.. *_makefile ------

# make build hs_makefile ...
for i in $my_makefiles ;do
	if [ "$build_makefile_flag" -eq "1" ] ; then
		echo  "gmake -f $i       | $wtee_to_log   " >>$all_bat_name 
	fi
done
# make clean statements #
echo " " >> $all_bat_name
if [ "$clean_flag" -eq "1" ]; then 
	for i in $my_makefiles ;do
		echo  "gmake -f $i clean | $wtee_to_log " >>$all_bat_name 
	done
fi

echo ":: clean up ::" >> $all_bat_name
for i in $my_makefiles ;do
	echo   "DEL /f $i | $wtee_to_log "  >>$all_bat_name
done


echo "pause" >> $all_bat_name
if [ "$save_flag" -eq "0" ]; then
	echo "del /f all_makefile_run.bat  ">>$all_bat_name #run bat file
fi
cat $all_bat_name
unix2dos $all_bat_name >/dev/null 2>&1   # use when copy to $win7/test.bat 
chmod 0700 $all_bat_name  # must give *.bat execute permission!!!
alias run_bat='sh $perl_p/run_bat.sh '

	run_bat $all_bat_name   #call all_makefile_run.bat





