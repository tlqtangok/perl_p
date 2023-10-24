#!perl
use Cwd; 
use File::Basename;
use feature qw(say);


my $bin = @ARGV[0]; 
die "- please input argv bin!\n" if $bin eq "" ;

my $OS =  $^O; 
my $raw_which = 'MAGIC_PATH';
if ($OS eq "cygwin" || $OS eq "linux"){
	$raw_which =  `which $bin`; 
}
elsif ($OS eq "MSWin32" ){
	my @where_list =  `where $bin`; 
    $raw_which = $where_list[0];
}
else {
	die "- error ! you OS is not linux or cygwin or win32 !\n"; 
}

chomp ( $raw_which ); 
if ($raw_which eq ""){
	die "- no found bin root, please run which bin !\n"; 
}

my $bin_dir = dirname($raw_which); 
chdir ($bin_dir); 


my $bin_root = &jd_getcwd(); 

if ($OS eq "MSWin32"){
	print " pushd " , $bin_root, "\n" ; 
}
else {
	print " cd " , $bin_root, "\n" ; 
}





### sub list ###
sub jd_getcwd(){
	my $cwd_ = getcwd();
	if ( $OS eq 'MSWin32' ){
		$cwd_ =~ s|\/|\\|g;
	}
	return $cwd_;
}


