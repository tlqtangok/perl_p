#!bash

### P4CLIENT LISTS ###
export home_p=/slowfs/us01dwslow025/arc_test/linqi
export ia_p=$home_p/ia_p

P4CLIENT_archi=architect_upload_installer
installer_path_archi=$home_p/ia_p4/$P4CLIENT_archi

P4CLIENT_mqx=mqx_installer
installer_path_mqx=$home_p/ia_p4/$P4CLIENT_mqx

P4CLIENT_mwdt=mwdt_upload_installer
installer_path_mwdt=$home_p/ia_p4/$P4CLIENT_mwdt

# $home_p/ia_p4/mqx_installer
######################

function   Usage(){
echo bash \$perl_p/architect_upload_installer.sh -d 'des_upload_archi_script' -T archi
exit
 }

 if  [[ "$#" -ne 4 ]]; then
         Usage
 fi
 while getopts 'd:t:T:?' argv ; do  # add :a, add more warning
         case $argv in
          d) des=$OPTARG && echo "-d -> $OPTARG" ;;
          T) target=$OPTARG && echo "-T -> $OPTARG" ;;
          t) target=$OPTARG && echo "-t -> $OPTARG" ;;
         \?) Usage ;;
 esac
done

if [ "$des" == "" ] || [ "$target" == "" ] ; then
        Usage
fi

	if [[ "$des" =~ "des_upload" ]]; then 
	echo "please write description "
	exit
	fi

if [[ "$target" =~ "mqx" ]]; then
        export P4CLIENT=$P4CLIENT_mqx
        export II=$installer_path_mqx
        pushd $II/ia_p

        p4 client -o  | grep -C1 dwarc
        p4 edit ./ia_[lw]*/[Rim]*
        yes|cp  $ia_p/mqx*/*w*/[Rim]* ./*win*/
        # yes|cp  $ia_p/mqx*/*l*/[iRa]*   ./*lin*/

        # p4 submit -d "$des"
        p4 submit -d "$des" && popd > /dev/null
	exit
fi
if [[ "$target" =~ "archi" ]]; then
        export P4CLIENT=$P4CLIENT_archi
        export II=$installer_path_archi
        pushd $II

        p4 client -o  | grep -C1 dwarc
        p4 edit $II/ia_[lw]*/[iRa]*
        yes|cp  $ia_p/archi*/*w*/[iRa]* ./*win*/
        yes|cp  $ia_p/archi*/*l*/[iRa]*   ./*lin*/

        p4 submit -d "$des" && popd > /dev/null
	exit
fi

if [[ "$target" =~ "mwdt" ]]; then
        export P4CLIENT=$P4CLIENT_mwdt
        export II=$installer_path_mwdt
        pushd $II/ia_p/mwdt*

        p4 client -o  | grep -C1 dwarc

        p4 edit ./ia_[lw]*/[Rim]*
        yes|cp  $ia_p/mwdt*/*w*/[Rim]*  ./*win*/
        yes|cp  $ia_p/mwdt*/*l*/[Rim]*  ./*lin*/

        p4 submit -d "$des" && popd > /dev/null
	exit
fi

echo "- -T MUST be one of : archi mqx mwdt " 
