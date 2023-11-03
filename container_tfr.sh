cd /root/jd/t
export _1arg=$1
export _2arg=$2
export _3arg=$3

if [[ "$_1arg" =~ "cd" ]];then
        export _1arg="tor"
        export _2arg="$_3arg"
fi

/usr/bin/tfr $_1arg $_2arg

