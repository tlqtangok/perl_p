#!bash 
# written by jd at 2018-09-28
# usage 
# ===
#   sh start_ngrok.sh 
#   sh start_ngrok.sh list 
#   sh start_ngrok.sh kill



cd `dirname \`readlink -f $0\``

mkdir -p t
export ff=/home/pi/jd/perl_p/ff.PL 
export fn_log=t/log.log

if [ "$1" = "" ]; then
	rm -rf $fn_log
	perl _start_ngrok.PL  >$fn_log 2>&1 &
fi 

if [ "$1" = "list" ]; then
	pgrep -al ngrok ; pgrep jupyter -al ; sudo df -k| grep 119
fi

if [ "$1" = "kill" ]; then
	pgrep perl -al |grep _start_ngrok  &&  pgrep perl -al |grep _start_ngrok | perl $ff 0 | xargs kill -9
	(pgrep -al ngrok &&  pgrep jupyter -al) && (pgrep -al ngrok ; pgrep jupyter -al) | perl $ff 0 | xargs kill -9 
fi

