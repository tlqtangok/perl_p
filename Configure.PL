#!perl

use feature qw(say); 

#say "perl";
#system('perl .ARChitect2_bat_hs_em_700_600_OK.PL');
do '.ARChitect2_bat_hs_em_700_600_OK.PL'; 
say "-";
###################
#---global Define #
###################
my %ver_arr=qw(
sp_eng4		_201403_SP1_eng4
RC3		_201403RC3
RC2		_201403RC2
RC1		_201403RC1
);

=pod


=cut
my $arch_ver=@ver_arr{RC3};  #version newer install

my $cmp_ver_0=@ver_arr{sp_eng4}; # cmp DIR_0

my $cmp_ver_1=@ver_arr{RC3};  #cmp DIR_1

###################

#---END Define
my @hs_bat,@em_bat,@_700_bat,@_600_bat; 
my @hs_mv,@em_mv,@_700_mv,@_600_mv; 
my @hs_cmp,@em_cmp,@_700_cmp,@_600_cmp;

my @all_bat_file, @mv_file,@cmp_file; 

my $perl_p =@ENV{"perl_p"};
chdir($perl_p);
my $file_all_bat='arch_bat/all_bat.bat';
my $file_mv_bld_p='.mv_bld_p.PL';
my $file_cmp_ver_0_1='.cmp_ver_0_ver_1.PL';
if ( !((-e $file_all_bat) && (-e $file_mv_bld_p) && (-e $file_cmp_ver_0_1))){
say "there must one file missed, check!"; exit ; 
}
@all_bat_file=`cat arch_bat/all_bat.bat`; 
@mv_file=`cat .mv_bld_p.PL`;
@cmp_file=`cat .cmp_ver_0_ver_1.PL`;

#--- main() --- #

sub say_array_to_file{
	my $basedir=join "",`pwd`;
	$basedir =~ s/\s+//g;
	$basedir=$basedir.'/	';
my ($FileName,@arr_to_write)=@_;
open(FP,">","$FileName");
my $log_filename=$basedir.$FileName;
say "Create $log_filename";

map{say FP "$_" if $_ ne "";}@arr_to_write;
}
close(FP);
#---
#get bat of hs, em ,700 , em
#---

@hs_bat[0]  =@all_bat_file[0]; 

@hs_bat[0]  =@hs_bat[0]; 
@em_bat[0]  =@hs_bat[0]; 
@_700_bat[0]=@hs_bat[0];
@_600_bat[0]=@hs_bat[0]; 
map{
push @hs_bat,$_ 	if m/HS/; 
push @em_bat,$_ 	if m/em\d/; 
push @_700_bat,$_   if m/ARC7\d/; 
push @_600_bat,$_   if m/6\d\d/ || m/AS/; 
}@all_bat_file; 
#
#---end get bat of ---
#

#
#--- get hs_mv script ---
#

my $i=0; 
@hs_mv  = @mv_file;
@em_mv  = @mv_file; 
@_700_mv= @mv_file; 
@_600_mv= @mv_file; 
map{
#--- Add arch_ver ---
if ( m/^[^\#]/ && m/ver=/ ){
	s/=.*/=\'$arch_ver\';/g; 
	@hs_mv[$i]=$_; 
	@em_mv[$i]=$_; 
	@_700_mv[$i]=$_; 
	@_600_mv[$i]=$_; 
#	say ; 
}; 
#--- Add Rename @arr_ : @p_hs ---
if(  m/NOT_RENAME_LIST/ ){
	@hs_mv[$i]   ='		@arr_=	@p_hs;' ; 
	@em_mv[$i]   ='		@arr_=	@p_em;' ; 
	@_700_mv[$i] ='		@arr_=	@p_700;' ; 
	@_600_mv[$i] ='		@arr_=	@p_600;' ;
}



$i++; 
}@mv_file; 
#--- get hs_mv, em_mv... #

#--- get cmp_file ---

@hs_cmp = grep(!m/ARC7/ && !m/em\d/ && !m/(ARC6)|(625x)|(AS\d)/,@cmp_file); 
@em_cmp = grep(!m/HS/ && !m/(ARC6)|(625x)|(AS\d)/ && !m/ARC7\d/,@cmp_file); 
@_700_cmp=grep(!m/HS/ && !m/(ARC6)|(625x)|(AS\d)/ && !m/em\d/,  @cmp_file); 
@_600_cmp=grep(!m/HS/ && !m/ARC7\d/ && !m/em\d/, 			@cmp_file);


#END get @hs_cmp
# ---end main---#


# test area ---
map{
my $_tmp;

$_tmp=$_;
$_=@hs_cmp[$_tmp];
chomp;
s/ver_0=.*/ver_0=\"$cmp_ver_0\";/g ;
s/ver_1=.*/ver_1=\"$cmp_ver_1\";/g;
s/template/hs_template/g;
@hs_cmp[$_tmp] = $_; 

$_=@em_cmp[$_tmp];
chomp;
s/ver_0=.*/ver_0=\"$cmp_ver_0\";/g ;
s/ver_1=.*/ver_1=\"$cmp_ver_1\";/g;
s/template/em_template/g;
@em_cmp[$_tmp] = $_; 

$_=@_700_cmp[$_tmp];
chomp;
s/ver_0=.*/ver_0=\"$cmp_ver_0\";/g ;
s/ver_1=.*/ver_1=\"$cmp_ver_1\";/g;
s/template/_700_template/g;
@_700_cmp[$_tmp] = $_; 

$_=@_600_cmp[$_tmp];
chomp;
s/ver_0=.*/ver_0=\"$cmp_ver_0\";/g ;
s/ver_1=.*/ver_1=\"$cmp_ver_1\";/g;
s/template/_600_template/g;
@_600_cmp[$_tmp] = $_; 

}(0..@_600_cmp*2);


# --- test area ---

my @dir_name =qw(
 dir_hs_cmp
 dir_em_cmp
 dir_700_cmp
 dir_600_cmp
);
# do RM
 
map{
chdir($perl_p); 
`rm -rf $_`;
`mkdir -p $_`;
chdir($_);
if (m/hs/) {
say_array_to_file('hs_bat.bat',@hs_bat);
say_array_to_file('hs_mv.PL',@hs_mv); 
say_array_to_file('hs_cmp.PL',@hs_cmp); `chmod 755 *.*`;
say "-";
	}
elsif (m/em/) {

say_array_to_file('em_bat.bat',@em_bat);
say_array_to_file('em_mv.PL',@em_mv);
say_array_to_file('em_cmp.PL',@em_cmp);`chmod 755 *.*`;

say "-";
}
elsif (m/700/) {

say_array_to_file('_700_bat.bat',@_700_bat);
say_array_to_file('_700_mv.PL',@_700_mv);
say_array_to_file('_700_cmp.PL',@_700_cmp);`chmod 755 *.*`;

say "-";
	}
else {

say_array_to_file('_600_bat.bat',@_600_bat);
say_array_to_file('_600_mv.PL',@_600_mv);
say_array_to_file('_600_cmp.PL',@_600_cmp);`chmod 755 *.*`;
say "-";
	}




}@dir_name; 

