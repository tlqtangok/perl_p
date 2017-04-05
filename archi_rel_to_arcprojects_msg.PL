#!perl
#BEGIN{@ENV{perl_p_}='/remote/us01home41/linqi/perl_p';push @INC,@ENV{perl_p_};}
BEGIN{push @INC,@ENV{perl_p};}
use feature qw(say);
use Jidor qw(Jidor_Usage subst_dot subst_foreach smart_match);
use strict;
# -
#  &subst_dot( @arr );                   ;# @arr return to @arr; 
#  &subst_foreach ( $file_name, @arr )   ;# return $cmd is the perl -i.bak -pe ...
#  &smart_match( $tag, @arr_num_or_str ) ;# -1 not match,or $loc_i is return 
# -
#---END Usage 


##########################################################
############ REPLACE OLD VERSION STRING ##################
##########################################################

if (!  defined @ARGV[0] ) {
	say ' perl $perl_p/archi_rel_to_arcprojects_msg.PL I-2014.03-SP2-RC5 ';
	exit;
}
my $perl_p=@ENV{perl_p};
my $new_ver=@ARGV[0]  ;
my $file_name = $perl_p.'/upload_archi_msg_R0.txt'; 
`cp $file_name /tmp/tmp.txt`;
$file_name='/tmp/tmp.txt';
my $old_ver='I-2014.03-SP2-RC4';
my @arr_file=`cat $file_name`;
my @arr_old_new=($old_ver, $new_ver);
my $cmd = &subst_foreach( $file_name, @arr_old_new) ; 
#say $cmd;
system($cmd); 
#say `cat tmp.txt`;

##########################################################
################# REPLACE RC TO noRC #####################
##########################################################
#--- remove -RCx 
my $media_RC_rm=$new_ver;
if( ($media_RC_rm =~ m/RC/i) || ($media_RC_rm =~ m/eng/i) ) {
	$media_RC_rm =~ s|\-RC\d||i;
	$media_RC_rm =~ s|\-eng\d||i;
}
say $media_RC_rm;

if ( $media_RC_rm ne $new_ver){
	my @arr_RC_noRC=('I-2014.03-SP2',$media_RC_rm);
	my $cmd = &subst_foreach( $file_name, @arr_RC_noRC) ; 
	system ($cmd);
#say `cat tmp.txt`;
}


##########################################################
#####################  GEN REPORT  #######################
##########################################################
say "\n####### T H E   R E P O R T ######\n";
say `cat $file_name`;
say "####### E N D   R E P O R T ######\n\n";



##########################################################
################# MATCH TWO FILE_LIST  ###################
##########################################################
my @dir_path=`cat $file_name |grep -A2 -i source|grep -v -i source`;
my @create_list_should_be=`cat $file_name |grep -A5 create-List |grep -v create-List`;

chomp($dir_path[0]);
$dir_path[0] =~ s|^\s+\/|\/|;
my $mydir = $dir_path[0];
if (! -d $mydir ) {
	say "Warning:";
	say "- folder $mydir not exist !" ; 
}
else{
	my @arr_real=`ls $mydir* `;
	map{
		$_='    '.$_;
	}@arr_real;
#say @arr_rel,@create_list_should_be;
	@arr_real = sort (@arr_real); 
	@create_list_should_be = sort @create_list_should_be; 
	say "- MATCHED arr_real ~~ arr_should_be " if @arr_real ~~ @create_list_should_be ;
	if (! (@arr_real ~~ @create_list_should_be) ){\
#map{print "___".$_."___"."\n";}@arr_real; 
#print "\n"."##########################"."\n";
#map{print "___".$_."___"."\n";}@create_list_should_be; 

		say "- NOT MATCH :";
		say "real:\n",@arr_real,"\n","should_be:\n",@create_list_should_be;
		`rm -rf /tmp/tmp.txt /tmp/tmp.txt.bak`;
		exit;

	};


}




`rm -rf /tmp/tmp.txt /tmp/tmp.txt.bak`;

