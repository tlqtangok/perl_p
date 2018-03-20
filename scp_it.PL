#!perl

my $arg0=@ARGV[0];
my $pwd_=`pwd`;chmod($pwd_);

if ( @ARGV==0 ){
$arg0=$pwd_;
@ARGV[0]=$pwd_;
}

my $hostaddr = `hostname -I`; 
$hostaddr =~ s/ //g;
chomp($hostaddr);
my $user_name_host_name=`whoami`.'@'.$hostaddr;
$user_name_host_name =~ s/\n//g;

map{

my $abs_path=`readlink -f "$_"`;
chomp($abs_path);
chomp($abs_path);

my $print_out_scp_statement = ' scp -r '."$user_name_host_name:$abs_path ./";
chomp($print_out_scp_statement);
if ($print_out_scp_statement =~ m/\n/){
	print $print_out_scp_statement;
}
else{
	print $print_out_scp_statement."\n" ;
}

}@ARGV; 


