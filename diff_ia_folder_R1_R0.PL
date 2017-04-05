#!perl

BEGIN{push @INC,@ENV{perl_p};};
use feature qw(say);
use Jidor qw(Jidor_Usage subst_dot subst_foreach smart_match);
# -
#  &subst_dot( @arr );                   ;# @arr return to @arr; 
#  &subst_foreach ( $file_name, @arr )   ;# return $cmd is the perl -i.bak -pe ...
#  &smart_match( $tag, @arr_num_or_str ) ;# -1 not match,or $loc_i is return 
# -
#---END Usage 

my $file_name = 'txt.txt'; 
my @arr=qw( 1 2 3 2-2 3.3 4.4-9); 
#my $cmd = &subst_foreach( $file_name, @arr) ; 
#system($cmd);
#
################################################ 
my $DIR_R0 = @ARGV[1];
my $DIR_R1 = @ARGV[0];
my $tmpFile="/tmp/tmp.txt";
open my $FN,'>',$tmpFile;
if ( !$DIR_R0 || !$DIR_R1 ){
	say 'perl $perl_p/ia_file_diff_R1_R0.PL `pwd` $DIR_R0 '; 
	exit;
}
if (! -d $DIR_R0 ){
	say "- no exist folder:\n\t$DIR_R0";
	exit;
}


say "\nexport DIR_R1=$DIR_R1	;export DIR_R0=$DIR_R0";
#say "$DIR_R1\t $DIR_R0";

my @arr_R0 = `ls $DIR_R0 `;
if (@arr_R0 ~~ m/architect/i){

my $_1="MAGIC_STR"; 
#say @arr_R0;
	map{chomp; 
		if($_ eq "jre"){
			if ( -d "$DIR_R0/$_/linux"){
			$_1="diff -uwqr $DIR_R1/../xCAM\t/slowfs/us01dwslow025/arc_test/linqi/ia_p/archi_ia/src/xcam/xcam_linux/xCAM\n ";
			$_=$_1."diff -uwqr $DIR_R1/$_/linux\t$DIR_R0/$_/linux\n";}
		else {
			$_1="diff -uwqr $DIR_R1/../xCAM\t/slowfs/us01dwslow025/arc_test/linqi/ia_p/archi_ia/src/xcam/xcam_win32/xCAM\n ";
			$_=$_1."diff -uwqr $DIR_R1/$_/win32\t$DIR_R0/$_/win32\n";}
		
		}
		else{
			$_="diff -uwqr $DIR_R1/$_\t$DIR_R0/$_\n";}
	}@arr_R0;




}
# mw 
elsif ( @arr_R0 ~~ m/metaware/i){ 
	map{chomp;$_='java/jre8' if m/(java)|(jre)/i ; $_="diff -uwqr $DIR_R1/$_\t$DIR_R0/$_\n";$_=qq(\n) if m/cleanup_script/;}@arr_R0;
	 
}

elsif ( @arr_R0 ~~ m/nsim/i ){

	map{chomp; $_="diff -uwqr $DIR_R1/$_\t$DIR_R0/$_\n";}@arr_R0;

}


	say $FN @arr_R0;
	map{ s|$DIR_R0|\$DIR_R0|g;s|$DIR_R1|\$DIR_R1|g ;}@arr_R0;
	say "- begin running:\n @arr_R0";


close $FN;
#system( "sh $tmpFile|grep -v solaris|grep -v win32" );
system( "sh $tmpFile|grep -v solaris|grep -v -P \'\\\.ico\'|grep -v native_parser_example" );
# native_parser_example'

`rm $tmpFile`;


