#!perl 
use feature qw(say); 
use File::Basename ;    # get dirname suffix and path
use File::Spec ;
### block###
#--- GLOBAL VARs---
my $mqx_new_ver='';
#
sub cp_a6_mk2nsim(){
	my $a6_mk='./nsim/arca6_config.mk';
	`cp -f config.mk $a6_mk ` if (! -e $a6_mk);
	`chmod 0555 nsim/*.mk `;
}
sub get_os_type(){
	my $OS_TYPE_header="# OS_TYPE= XXX " ;
	if (`set|grep -i ostype` =~ m/linux/i || @ARGV[1] =~ m/linux/i ) {
		$OS_TYPE_header ="# OS_TYPE=Linux " ; 
	}
	elsif (@ENV{ComSpec}.@ENV{COMSPEC} =~ m/CMD.EXE/i ) {
		$OS_TYPE_header= "#    OS_TYPE=Windows    " ; 
	}  
	else {
		die "Couldn't figure out your OS_TYPE"; 
	}
#say $OS_TYPE_header; 
	return $OS_TYPE_header." CPU_ARCH=@ARGV[0] #\n"; 
}
sub get_mqx_configs_to_hash($){
	my ($OS_TYPE)=@_; 
	my %var_mqx=();
	my $processor =@ARGV[0];
	my $die_prompt= ' Usage:
	perl _*.PL a6|a7|em|hs [ linux | win ]
	' ;
	$processor eq "" && die $die_prompt ;
	$_=`pwd |xargs dirname `;
	s/\R//g; @var_mqx{MQX_ROOT}=$_ ;
	$mqx_new_ver=basename($_);
	$_= ` which ccac |xargs dirname |xargs dirname `; s/\R//g;@var_mqx{COMPILER_ROOT}=$_;
	$_=`ls nsim|grep .mk|grep -i $processor` ; s/\R//g;@var_mqx{full_cpu_config}=$_;  
	$_="@var_mqx{MQX_ROOT}/build/nsim/@var_mqx{full_cpu_config}";s/\R//g; @var_mqx{MQX_CONFIG}=$_;
	die " no $processor config.mk find in path @var_mqx{MQX_CONFIG} !\n " if !(@var_mqx{full_cpu_config}=~m/mk/i );
	delete @var_mqx{full_cpu_config};
	map{s|^.*\/c\/|c\:\\|g;s|\/|\\|g; }%var_mqx  if ($OS_TYPE =~ m/windows/i); 
#say %var_mqx ; 
	return %var_mqx;
}
sub get_hash_to_var($ \%){
	my ($OS_TYPE, $varmqx)=@_; 
	my %var_mqx=%$varmqx; 
	my @arr = keys %var_mqx;
	my $_MQX_CC_ROOT='export MQX_COMPILER_ROOT:=$(COMPILER_ROOT)'; 
	push @arr,$_MQX_CC_ROOT; 
	my $CONFIGS_STATEMENTS=$OS_TYPE;
	map{
	if (! m/=/) {
		$CONFIGS_STATEMENTS .= "export $_:=@var_mqx{$_}\n";
	}
	else {
		$CONFIGS_STATEMENTS .= "$_\n";
	}
	}@arr;
#say $CONFIGS_STATEMENTS ; 
	return $CONFIGS_STATEMENTS ; 
}#END get_hash_to_var
sub input_mqx_configs_into_makefile($ $){
	my ($fn,$CONFIGS_STATEMENTS) =@_; 
	`chmod 0755 $fn `;
	say "\n#	export MQX_CONFIGs statement	#\n";
	say $CONFIGS_STATEMENTS;
	my $BAK_fn='bak.'.$fn; 
# bak #
	if (! -e $BAK_fn ) {
		`cp $fn $BAK_fn `; 
		say "bakup $fn to $BAK_fn..."; 
	}
	`cp $BAK_fn $fn ` ; 
	my @arr_old=`cat $fn`;
	map{
	s/(\.PHONY)/$CONFIGS_STATEMENTS\n\1/g; 
#print ;  
	}@arr_old;
	open FP,'>',$fn;
	print FP @arr_old;
	close FP;
	say '#	verify if exist @arr= export in makefile	#'."\n";
	@arr_old=`cat makefile`;
	@arr_old=grep  m/(^export|\.phony)/i , @arr_old;
	say @arr_old; 
}
cp_a6_mk2nsim();
my $OS_TYPE=get_os_type(); 
my %mqx_var=get_mqx_configs_to_hash($OS_TYPE);
my $CONFIGS_STATEMENTS = get_hash_to_var($OS_TYPE, %mqx_var); 
input_mqx_configs_into_makefile("makefile", $CONFIGS_STATEMENTS); 

# config all lnk file under MQX_ROOT = c:\arc\software\mqx2.61
#perl $perl_p/lnk_mqx_new_ver.PL $mqx_new_ver 
#my $perl_p=@ENV{perl_p}.'/';
#system("perl $perl_p"."lnk_mqx_new_ver.PL $mqx_new_ver"); 
