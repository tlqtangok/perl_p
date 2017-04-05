
#cat /qftest/jd_qf_test_0.qft -n |grep -v _name\=\"|grep -v \#|grep -v rc.|grep procedure_|grep -v '('
SRC=$qft_mide/common/mide_api.qft
cat ${SRC} -n |grep -v _name\=\"|grep -v \#|grep -v rc.|grep procedure_|grep -v '(' |grep name= |perl -e '@arr=<>;map{chomp;@a=split m/name\=/;@a[-1]=~s/\W//g;print @a[-1]."\n"; }@arr; ' 
