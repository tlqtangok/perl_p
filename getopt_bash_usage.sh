#!/bin/bash

# sh  1.sh t0 t1  -a  -b bv
cat <<-EOF


EOF
#echo $@
 
#-o或--options选项后面接可接受的短选项，如ab:c::，表示可接受的短选项为-a -b -c，其中-a选项不接参数，-b选项后必须接参数，-c选项的参数为可选的
#-l或--long选项后面接可接受的长选项，用逗号分开，冒号的意义同短选项。
#-n选项后接选项解析错误时提示的脚本名字
ARGS=`getopt -o ab:c:: --long along,blong:,clong:: -n 'example.sh' -- "$@"`
if [ $? != 0 ]; then
    echo "Terminating..."
    exit 1
fi
 
#echo $ARGS
#将规范化后的命令行参数分配至位置参数（$1,$2,...)
eval set -- "${ARGS}"

export X=abc
while true
do
    case "$1" in
        -a|--along) 
            echo "Option a";
			cat <<-EOF
			show help \\
			info \\
			ok	
	EOF
            shift;
            ;;
        -b|--blong)
            echo "Option b, argument $2";
			export X=$2
            shift 2
            ;;
        -c|--clong)
            case "$2" in
                "")
                    echo "Option c, no argument";
                    shift 2  
                    ;;
                *)
                    echo "Option c, argument $2";
                    shift 2;
                    ;;
            esac
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Internal error!"
            exit 1
            ;;
    esac
done
 
#处理剩余的参数
for arg in $@
do
    echo "processing $arg"
done

echo $1
echo $2

echo x is $X
