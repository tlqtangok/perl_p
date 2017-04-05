#!perl

use strict; 
### Usage ###
# perl this_perl_name.PL  create build dbg... 
# ###########
my $fun_pattern1=@ARGV[0];
my $fun_pattern1=@ARGV[1]; 
my $fun_pattern2=@ARGV[2]; 
my $fun_pattern3 =@ARGV[3]; 
my $fun_pattern4 =@ARGV[4]; 
my $fun_pattern5 =@ARGV[5]; 

my $fun_pattern0=@ARGV[0]; 

if ($fun_pattern0 eq ""){
	$fun_pattern0='\w';
	print "Usage:\n\tperl qft_code_gen.PL {create,build,dbg,run,ide}\n"; 
	exit(); 
}
my @arr=`sh qftest.sh`;
#print @arr; 
#<Procedure id="_1oTJ" name="id_procedure_dbg_current_project_no_p0">
my $templ_full='
id_procedure_name="id_procedure_create_project_with_p0_id_core_family_name_p1_id_tcf_name_return_id_project_name"
id_core_family_name="em"
id_tcf_name="em5d_mini.tcf"
xxxxx=rc.callProcedure(id_procedure_name,{"id_core_family_name":id_core_family_name,"id_tcf_name":id_tcf_name})
';
my $templ_simple='
id_procedure_name="id_procedure_build_current_project_no_p0"
#simple_call(id_procedure_name)
';

my %hash_ans=();

map{

my $str_ans="";
chomp;
my @arr_1=split m/name\=/, $_;
@arr_1[-1] =~ s/\W//g;
$_=@arr_1[-1];
my $str_R0=$_;
#print '$str_R0:  ' , "$str_R0", "\n";
#print "### $str_R0 ###\n"; 
$str_ans.= "### $str_R0 ###\n"; 
my $flag_return_=0;
my $return_str="";
$flag_return_=1 if m/_return_/;
my @arr_2=split m/_p\d_|_return_/;
#print "@arr_2", "\n"; 
if ($flag_return_==1){
	$return_str=@arr_2[-1];
	@arr_2=@arr_2[1..@arr_2-2];
}
else {
	@arr_2=@arr_2[1..@arr_2-1]; 
}

$str_ans.= 'ID_P="'.$str_R0."\"\n"; 

my $i=0;
for($i=0; $i<@arr_2; $i++ ){
	$str_ans.= @arr_2[$i].q( = ) .q("").qq(\n); 
}#end for $i

# create param list dic 
#{"id_core_family_name":id_core_family_name,"id_tcf_name":id_tcf_name}
my $id_param_list="";
if (@arr_2>0){
	for($i=0; $i<@arr_2; $i++ ){
		$id_param_list.=q(")."@arr_2[$i]".q(").q(:)."@arr_2[$i]"."," ;
	}#end for $i
}
else {
	$id_param_list="";
}
$id_param_list=q({).$id_param_list.q(});

# get call statement
#xxxxx=rc.callProcedure(id_procedure_name,{"id_core_family_name":id_core_family_name,"id_tcf_name":id_tcf_name})

if ( $flag_return_==1){
	$str_ans.= $return_str.' = '.'rc.callProcedure(_H_+ID_P,'.$id_param_list.')'."\n"; 
}
else{
	$str_ans.= 'rc.callProcedure(_H_+ID_P,'.$id_param_list.')'."\n"; 
}

#print "\n";
$str_ans.= "\n";


@hash_ans{$fun_pattern0}=$str_ans   if $str_ans =~ m/$fun_pattern0/m && $fun_pattern0 ne ""; 
@hash_ans{$fun_pattern1}=$str_ans   if $str_ans =~ m/$fun_pattern1/m && $fun_pattern1 ne "";
@hash_ans{$fun_pattern2}=$str_ans   if $str_ans =~ m/$fun_pattern2/m && $fun_pattern2 ne "";
@hash_ans{$fun_pattern3}=$str_ans   if $str_ans =~ m/$fun_pattern3/m && $fun_pattern3 ne "";
@hash_ans{$fun_pattern4}=$str_ans   if $str_ans =~ m/$fun_pattern4/m && $fun_pattern4 ne "";
@hash_ans{$fun_pattern5}=$str_ans   if $str_ans =~ m/$fun_pattern5/m && $fun_pattern5 ne "";

}@arr;

my $_H_ = '../../common/mide_api.qft#'; 
my $h_statement = "_H_" . " = " . "'$_H_'"; 
print $h_statement, "\n"; 
print @hash_ans{$fun_pattern0}, @hash_ans{$fun_pattern1}, @hash_ans{$fun_pattern2}, @hash_ans{$fun_pattern3};
print @hash_ans{$fun_pattern4}, @hash_ans{$fun_pattern5}; 




