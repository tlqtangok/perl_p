#!bash
pwd| perl -e ' $_=<>;chomp($_);$cmd=q(export ARCHITECT_ROOT=$_)."/ARChitect".q( && export PATH=${ARCHITECT_ROOT}/bin/linux:).$_.q(/xCAM/i686-RHEL4-gcc-3.2.3/bin).q(:${PATH} && export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ARCHITECT_ROOT}/lib/linux); $cmd =~ s/\$_/$_/; print $cmd."\n"; '

