#!perl
use FindBin qw($Bin);
#print $Bin
use strict ;
use File::Basename;
use Cwd;
use Data::Dumper;

#chdir('test');
#print `pwd`;


#print $Bin;

my $const = q(
/slowfs/us01dwslow025/IDE_DAILY_BUILDS dialy builds 
/slowfs/us01dwslow025/mide/IDE_BUILD_RELEASES each check in 
	from http://arcjenkinsdev/job/IDE_BUILD_RELEASE/  jenkins for mide each build 

/slowfs/us01dwslow025/mide/MIDE_DSF_Checkin  ide_dsf build 
	from 
); 


### may need to be edit ###
my $DAILY_MIDE = '/slowfs/us01dwslow025/arc_test/linqi/tmp/ide_daily/ide'; 
my $LATEST_ECLIPSE_DIR = '/slowfs/us01dwslow025/mide/IDE_BUILD_RELEASES'; 



my $eclipse_suffix = 'products/com.arc.ide/linux/gtk/x86'; 
my $linqi_perl_p = '/remote/us01home41/linqi/perl_p'; 
my @arr_latest_eclipse_dirname = `perl $linqi_perl_p/lsh.PL $LATEST_ECLIPSE_DIR `; 

my $latest_eclipse_dirname = @arr_latest_eclipse_dirname[0]; 
$latest_eclipse_dirname =~ s/^\s*//; 
chomp( $latest_eclipse_dirname ); 

my $latest_eclipse_full_path = $LATEST_ECLIPSE_DIR .'/'. $latest_eclipse_dirname .'/'. $eclipse_suffix ; 

my $new_xml = "$latest_eclipse_full_path/artifacts.xml"; 
my $old_xml = "$DAILY_MIDE/eclipse/artifacts.xml"; 

my $new_eclipse = $latest_eclipse_full_path; 
my $old_eclipse = "$DAILY_MIDE/eclipse"; 


print "- latest check in is:  $latest_eclipse_dirname\n"; 
 


if ( ! -e  $new_xml ){
die "- no exist $new_xml !\n"; 
}

#print $latest_eclipse_full_path."___"; 

#print $new_xml, "\n", $old_xml,"\n"; 

if ( -e $old_xml ){
# diff xml files

my @diff_result_xml = `diff  $new_xml $old_xml` ; 
if (@diff_result_xml == 0){
# donothing , already latest mide 

print "- $old_eclipse is already latest version\n"; 
}
else {
# do delete and copy 
#print "- do delete old eclipse  $old_eclipse\n"; 
	system( &delete_old_eclipse($old_eclipse) ); 
#print "- do copy new eclipse $new_eclipse\n"; 
	system( &copy_eclipse_to_new_eclipse($new_eclipse, $old_eclipse) ); 

}

}
else{
# do copy
#print "- do copy new eclipse $new_eclipse\n"; 
	system( &delete_old_eclipse($old_eclipse) ); 
}


# mk log in $DAILY_MIDE #
my $log_file_dir = dirname $DAILY_MIDE;

system ( " echo $latest_eclipse_dirname > $log_file_dir/$latest_eclipse_dirname.txt " ); 

### SUB LIST ###
sub delete_old_eclipse($old_eclipse_dir){
my $old_eclipse_dir = @_[0]; 

if ( (basename $old_eclipse_dir) ne "eclipse"){
die "- old_eclipse_dir $old_eclipse_dir error!\n";
}
my $cmd_delete = qq( chmod -R 0777 $old_eclipse_dir && rm -rf $old_eclipse_dir ); 
return $cmd_delete."\n" ; 
}
sub copy_eclipse_to_new_eclipse($from_, $to_){
my ($from_, $to_) = @_; 
my $cmd_copy = qq( mkdir -p $to_ && cp -r $from_/. $to_/ ); 
return $cmd_copy ."\n"; 
}

# END   





