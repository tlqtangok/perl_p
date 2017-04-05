#!perl
use strict;
#find_by_name '*'| perl _perl.PL
use feature qw(say);
use Getopt::Std;

sub if_file_match_pat{

	my ($filename, $pat)=@_;
	my $FP; 
	open $FP, "$filename";
	while(<$FP>){
		return q(true) if m/$pat/; 


	}
	return qq(false);
	close $FP;


}
sub Usage{
	my %options=();
	getopts("?",\%options);  
	if (defined @options{'?'}){
		say q(Usage :
		find_by_name '*'| perl *.PL
		);
		exit 1; 
	}
}
sub get_basename_mqx2_61(){
	my $ret_mqx261=` pwd | xargs basename `;
	chomp $ret_mqx261;
	$ret_mqx261 =~ s|\.|\\\.| ; 
	return $ret_mqx261;
}
sub subst_each_mqx260_to_mqx261($){

	my ($new_mqx_ver)=@_;

	my $old_mqx_ver=q(').q(s/mqx[2-9]\.\d{2}[a-g]?/);
	my $old_mqx_ver_pat=q(mqx[2-9]\.\d{2}[a-g]?);
# with $BAK_FLAG , then gen .bak .
	my $BAK_FLAG='-i=subst on original file' ; 
#	my $BAK_FLAG='-no_subst' ; 

	my @arr=<>;
	my $FP_sh;
	open $FP_sh,'>tmp.sh';

	say "subst list :";
	while(@arr){
		$_=shift @arr;
		chomp;
#	my $filecontent = `cat "$_" `;
		my $filename = "$_"; 
		if( -T $filename &&  ! ($filename =~ m/\.bak/ms) && if_file_match_pat($filename, $old_mqx_ver_pat) eq "true" ){
			say $filename; 
			my $perl_lang='';
			$perl_lang=q(perl -i -pe ).$old_mqx_ver.$new_mqx_ver.q(/g;' ').$filename.q(') if  $BAK_FLAG =~ m/i/; 
			$perl_lang=q(perl    -pe ).$old_mqx_ver.$new_mqx_ver.q(/g;' ').$filename.q(') if !$BAK_FLAG =~ m/i/; 
			say $FP_sh $perl_lang; 
			my $rm_mqx_bak="rm ".q(').$filename.q(.bak).q(');
			say $FP_sh $rm_mqx_bak;
		}
	}
# --- for ultimate release ----

	say $FP_sh "\n"x2;
	say $FP_sh ' # - I will RM all the *bak* files ';
	say $FP_sh ' alias find_by_name='.q('find ./ -name ');
	say $FP_sh ' find_by_name '.q('*bak*').q(|xargs rm -rf ) ;   # clean all the bak file up ; 


	close $FP_sh;
	say "\n-clear up *bak* recrusively, rm this running *.sh ...";
	system('sh tmp.sh && rm *sh 2>/dev/null' );

}
#---main()---
Usage(); 
my $new_mqx261=get_basename_mqx2_61(); 
subst_each_mqx260_to_mqx261($new_mqx261);
