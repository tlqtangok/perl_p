#!perl
#use feature qw(say);
use strict;

package Jidor;
require Exporter; 
our @ISA=qw(Exporter);

#use Jidor qw(Jidor_Usage subst_dot subst_foreach smart_match);
our @EXPORT=qw(Jidor_Usage subst_dot subst_foreach smart_match);

#--- Usage: 
# !perl
# BEGIN{push @INC,@ENV{perl_p};}
# BEGIN{@ENV{perl_p_}='/remote/us01home41/linqi/perl_p';push @INC,@ENV{perl_p_};}
# use Jidor qw(Jidor_Usage subst_dot subst_foreach smart_match);
# -
#  &subst_dot( @arr );                   ;# @arr return to @arr; 
#  &subst_foreach ( $file_name, @arr )   ;# return $cmd is the perl -i.bak -pe ...
#  &smart_match( $tag, @arr_num_or_str ) ;# -1 not match,or $loc_i is return 
# -
#---END Usage 






sub Jidor_Usage(){
	my $perl_pm_dir = @ENV{perl_p}.'/';
	my $Jidor_pm = 'Jidor.pm';
	my $pm_full_path =$perl_pm_dir.$Jidor_pm;
	my @arr=`cat $pm_full_path |grep -A14 -P 'Usage\:' `; 
	print  @arr, "\n"; 
}
sub subst_dot(\@){
#	my ($arr_str) = \@_;
	map{
	chomp;
	s|\\|\\\\|g;
	s|\/|\\\/|g;
	s|\.|\\\.|g;
	s|\*|\\\*|g;
	s|\?|\\\?|g;
	s|\-|\\\-|g;
	}@_;
return @_; 
#print "@_";

} #end subst_dot();

# @arr = ("str1","str1_new","STR2.2","STR3.3";
# subst_foreach($file_name, @arr) # s/@arr[0]/@arr[1]/g;
sub subst_foreach($ \@){

	my ($file_name, @arr_str) = @_;
	my @arr=@arr_str; 
	my $len=@arr; 
#	say " len of arr is $len"; 
	if(@arr%2 == 1){
		print "- error subst str list ! ", "\n";
		exit(1);
	}
	&subst_dot(@arr);
# print "@arr";
	my $i=0;
	my $cmd_total='';
	my $arr_len=@arr;
	map{
	$cmd_total .= q(perl -i.bak -pe ' ) if $i == 0 ;

#$cmd_total .= q(s/@arr[).$i.q(]/@arr[).($i+1).q(]/ ; );
	$cmd_total .= qq(s/@arr[$i]/@arr[($i+1)]/g; );
	$cmd_total .= qq(' $file_name ) if $i == ($arr_len-2);
	$i=$i+2;
	}(0..$arr_len/2-1);
#	say "$cmd_total";      # if you want to display the running cmds. dis-comment this line.
#	system($cmd_total);
	return $cmd_total;
} # end subst_foreach

sub smart_match($ \@){
	my ($tag,$arr_str) = @_;
	my	@arr=@$arr_str;
	my $i=0;
	for( $i=0;$i<@arr;$i++){
		last if( $tag == @arr[$i] || $tag eq @arr[$i] ) ;

	}#end for()
	return -1 if ($i == @arr);
	return $i;

}


