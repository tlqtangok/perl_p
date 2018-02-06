#!bash
export CUR_DIR=`pwd`
if [ "$1" != "" ]; then
        export CUR_DIR=$1
fi
perl -e ' $_ = @ENV{CUR_DIR}; if (m/\/[c-z]\//){s|/([c-z])\/|\1\:\\|i ; s|\/|\\|g ; print " ".$_ ;} else { $d=$_;my $where_git = `where git|grep cmd.git -i `; $where_git =~ s/.cmd.git.*$//i; chomp($where_git);$d=~s|\/|\\|g ; print qq($where_git).$d } '



