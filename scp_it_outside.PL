#!perl

my $arg0=@ARGV[0];
my $pwd_=`pwd`;chmod($pwd_);

if ( @ARGV==0 ){
$arg0=$pwd_;
@ARGV[0]=$pwd_;
}

my $hostaddr = `hostname -I`; 
my @host_arr_maybe = split m/\s+/, $hostaddr;
@host_arr_maybe[0] =~ s/ //g;
$hostaddr = @host_arr_maybe[0];
my $algo_addr = join ".", reverse qw(14 3 108 39);
$hostaddr = $algo_addr;
my $port = '10247'; 

chomp($hostaddr);
my $user_name_host_name=`whoami`.'@'.$hostaddr;
$user_name_host_name =~ s/\n//g;

map{
chomp($_); 
my $abs_path=`readlink -f "$_"`;
chomp($abs_path); 

chomp($abs_path); 


say STDERR "- error! exist path or file : $abs_path\n" if ! -e $abs_path; 

chomp($abs_path);
chomp($abs_path);

# if we wanna add step machine here 
# my $_O_paras =  q{ -o "ProxyCommand ssh -q -p 22 tanglq@step_machine_ip  -W %h:%p" }; 
# my $print_out_scp_statement = " scp $_O_paras -P $port -r "."$user_name_host_name:$abs_path ./";

my $_R_flag = "";
$_R_flag = "-r" if -d $abs_path;

my $print_out_scp_statement = " scp -P $port $_R_flag "."$user_name_host_name:$abs_path ./";

chomp($print_out_scp_statement);
if ($print_out_scp_statement =~ m/\n/){
	print $print_out_scp_statement;
}
else{
	print $print_out_scp_statement."\n" ;
}

}@ARGV; 


