#!perl
#BEGIN{@ENV{perl_p_}='/remote/us01home41/linqi/perl_p';push @INC,@ENV{perl_p_};}
#BEGIN{push @INC,@ENV{perl_p};}
#use feature qw(say);
#use Jidor qw(Jidor_Usage subst_dot subst_foreach smart_match);
# -
#  &subst_dot( @arr );                   ;# @arr return to @arr;
#  &subst_foreach ( $file_name, @arr )   ;# return $cmd is the perl -i.bak -pe ...
#  &smart_match( $tag, @arr_num_or_str ) ;# -1 not match,or $loc_i is return
# -
#---END Usage

#my $file_name = 'txt.txt';
#my @arr=qw( 1 2 3 2-2 3.3 4.4-9);
#my $cmd = &subst_foreach( $file_name, @arr) ;
#system($cmd);
#
# lsh -A $dir_name
my $cmd_templ='ls -thl -A \'$dirname_\' '; 
my $pwd_=`pwd`; 
chomp($pwd_); 
use Getopt::Std;
%options=();
getopts("A:",\%options);   # if with : then must need a param_value
my $arg0='MAGIC_STR_ARG0';

my @A_keys_of_options= keys %options; 
if (! ("@A_keys_of_options" =~ m/A/) ){
	$cmd_templ =~ s/\-A//;

	if(@ARGV[0] eq ""){

		$cmd_templ =~ s/\$dirname_/$pwd_/; 

	}
	else {
$arg0=`readlink -f @ARGV[0]`;chomp($arg0);  
		$cmd_templ =~ s/\$dirname_/$arg0/; 
	}

}
else {
#	say "else" ; 
my $dirname=@options{A}; 
	if( @options{A} eq ""){
		$cmd_templ =~ s/\$dirname_/$pwd_/; 
	}
	else {
		$arg0=`readlink -f @options{A}`; chomp($arg0); 
		$cmd_templ =~ s/\$dirname_/$arg0/; 
	}
}

#say $cmd_templ; 
#add chinese support :
my @cmd_result=`$cmd_templ|grep -v -i total | grep -v \'总用量 \' `; 
map{
@cmd_result[$_] =~ s/^.*\s(\d\d\:?\d\d)\s/ /g; 
print @cmd_result[$_];
}(0..6);



