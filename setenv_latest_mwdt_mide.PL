#!perl 
use strict ; 
### use this script to auto-sync latest mwdt + latest mide ###

# change the $JAVAROOT to where you have write access permission

### ***VERY IMPORTANT*** ###
# and then cp java folder from latest mwdt to $JAVAROOT ,only need to be done *ONCE*
### 


############ GLOBAL VAR ############

# edit this line only #
my $JAVAROOT 	= '/SCRATCH/ARC/ARC_'  ; 	# a location where you have writing permision


my $latest_mide = '/slowfs/us01dwslow025/ide_builds/latest/Linux'; 	# latest_mide
my $latest_mwdt = '/slowfs/us01dwslow025/mwdt_builds/latest';		# latest_mwdt
####################################




my $perl_p = '/remote/us01home41/linqi/perl_p'; 	# some libraries are put here. 
my $cmd=q(
ide_ver=$latest_mide_ver
export JAVAROOT=$JAVAROOT
mkdir -p $JAVAROOT/MetaWare
chmod -R  0775  $JAVAROOT/MetaWare && rm -rf $JAVAROOT/MetaWare/ide  &&  cd $JAVAROOT/MetaWare/ && cp -r $latest_mide/../../${ide_ver}/Linux/ide ./
); 

# get version of latest mide
my $latest_mide_ver = `readlink -f  $latest_mide`; 
$latest_mide_ver =~ s/.*\/(ide.20\d\d.*\d\d\d)\/.*/\1/; 
chomp($latest_mide_ver); 

### custom the cmd ###
$cmd=~s/\$latest_mide_ver/$latest_mide_ver/; 
$cmd =~ s/\$latest_mide/$latest_mide/; 
$cmd =~ s/\$JAVAROOT/$JAVAROOT/mg; 

# gain mwdt env var from setenv_mwdt.sh 
my $source_src_latest_mwdt = qq( cd   $latest_mwdt/linux/ARC).q( && sh *.sh 1>~/txt.txt 2>/dev/null  ); 
$cmd=$cmd."\n". $source_src_latest_mwdt; 

# run it and console
print "- run sustitute $JAVAROOT/ide\n";
system( " readlink -f $latest_mwdt ") ; 
system(" readlink -f $latest_mide "); 
system($cmd); 
print "- please run:\n source ~/txt.txt && export PATH=$JAVAROOT/MetaWare/ide:\$PATH\n"; 
#print "\t or run \"s_l\"\n" ; 

