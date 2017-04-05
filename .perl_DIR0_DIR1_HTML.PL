#!perl
use 5.010_001;
#use Time::Tiny;
#STDIN array say print grep map localtime println()
#use autodie;
#use IPC::System::Simple;

#use Jidor qw(func_1 func_2);
use List::Util qw(first sum max shuffle maxstr);
=pod

=cut

use Text::Diff::FormattedHTML;
# as you might want some CSS:
#######################################################################################
sub println{
say "\n---------------------";
};
$flag_first_output=0;
	
	sub process_line{
	$_=@_[0];
	$_="TIME IS 20xx-xx-xx\n" if m/ 201\d/ ;
	s/\d{13}\"/modified_time_is\"/g;
	s/link_\d{1,3}/link_id/g;
	s/globalInstance=\"\d{1,3}\"/globalInstance=\"id\"/g; 
	s/.*I-201[34]\..*//g;
		s/version=\"[\d\D]{5,20}\"/version=\"version_No\.\"/g; 
		s/checksum=\"\w{0,33}\"/checksum=\"checksum_No\.\"/g; 
	return $_; 
	}


sub file1_vs_file2{
	@arr_params=@_;
my ($file_0,$file_1,$OUT)=@arr_params;
if($file_0 && $file_1){
say "vimdiff $file_0  $file_1";

#header of CSS ---
print $OUT "<style type='text/css'>\n", diff_css(), "</style>\n" if $flag_first_output==0;


$flag_first_output++; 

open $f0,'<',"$file_0";
my @arr_f0=<$f0>;
close($f0); 
#---ignore the time info---

map{
$_=process_line($_);

}@arr_f0;

open $f1,'<',"$file_1";
my @arr_f1=<$f1>;
close($f1); 
map{
$_=process_line($_);
}@arr_f1;



if(@arr_f0 ~~ @arr_f1){
	#do nothing; 
}
else  {
print $OUT "<br><font color='red'  size=\"5\"> $file_0</font>          .vs.         <font color='red' size=\"5\"> $file_1</font><br>  ";
print $OUT diff_strings( "@arr_f0","@arr_f1");
}

return true;

}#---end if $file_0 is valid---


else {return "NULL"; }; 
} #end sub file_1_vs_file2
############################################################################################
#
#
#
#
#---change directory---
$my_hs_p=@ENV{hs_p}; 

$mydir=chdir $my_hs_p;

#---txt.txt , generate from diff -r DIR1 DIR2 |grep '^diff -rc' ---
say `rm -vrf tmp_*.txt`;
#$time_id=$$;

#---DIR_NAME_0 DIR_NAME_1---
if(defined(@ARGV[0]) && defined(@ARGV[1])){
$DIR_0=@ARGV[0]; 	$DIR_1=@ARGV[1];
}
else{
$DIR_0='p_hs_0904'; 
$DIR_1='p_hs_0903';
}; 
$save_txt="tmp_$$.txt"; 

say "writing diff_files to $save_txt...\n"; 
`diff -r $DIR_0 $DIR_1|grep \^diff >$save_txt`;
say "writing finished !\n";


#---
$files=$save_txt;  
$show_files=`cat $files`;
#say "show files is :\n $show_files  \n"; 
#`rm -rf *.html`;
open IN, '<', "$files";
@all_files=<IN>;
close(IN);
#--say @all_files---
#say @all_files; 
#---RM .html RO---

$HTML_FILE="diff_"."$DIR_0"."_vs_"."$DIR_1".'.html';

$RM_HTML=`rm -vrf *.html`; 
say $RM_HTML; 
open $FP_HTML, ">>$HTML_FILE"; 

map{
chomp; 
s/^diff -r //g; 
s/.*log.*//g;
s/.*\.html\z//g;
s/ $DIR_1/:$DIR_1/g;  #--- DIR_0 DIR_1; 
chomp; 
my($file_0,$file_1)=split /:/;
file1_vs_file2($file_0,$file_1,$FP_HTML); 

}@all_files;
#say @all_files; 
`mv -f $HTML_FILE Diff_HTML_FILES/$HTML_FILE`;

close($FP_HTML); 
