#!perl
use feature qw(say);
use File::Basename ; # get dirname suffix and path
use File::Spec ; # to get file's abs path; 
use File::pushd;  #must  my dir = pushd($dir0); 
use Cwd;     #getcwd == `pwd` 
use Win32::Shortcut;

sub print_lnk($){
	my ($LINK)=@_;
	print join("\n", "Path--- ".$LINK->Path,
		"ShortPath--- ".$LINK->ShortPath,
		"Arguments--- ".$LINK->Arguments,
		"WorkingDirectory--- ".$LINK->WorkingDirectory,
		"Description--- ".$LINK->Description,
		"ShowCmd--- ".$LINK->ShowCmd,
		"HotKey--- ".$LINK->Hotkey,
		"IconLocation--- ".$LINK->IconLocation,
		"IconNumber--- ".$LINK->IconNumber)."\n";
}

sub get_mqx_root_from_build_dir($){
	my ($build_dir)=@_;
	chomp($build_dir);
	my $mqx_root=dirname($build_dir);
	return $mqx_root;
}#--- end get_mqx_root_from_build_dir($build_dir)

sub process_each_lnk($ $){
	my ($LINK_path, $mqx_new_ver)=@_;
	chomp($LINK_path);chomp($mqx_new_ver);
	my $LINK=new Win32::Shortcut();
	$LINK->Load($LINK_path) or die "$LINK_path not found!\n";
	my $regex_old_mqx_ver=qr(mqx[2-9]\.\d{2}[a-g]?); #must in one line

	my $log = "\n---Verify---\nprocess_each_lnk [$LINK_path]\nlnk WorkingDirectory\n"; 
	$log.=" RO: $LINK->{WorkingDirectory}\n";
#	`rm "$LINK_path" `;  # prevent the write permission puzzles 

	if($mqx_new_ver =~ m|$regex_old_mqx_ver|){
		$LINK->{'WorkingDirectory'} =~ s|$regex_old_mqx_ver|$mqx_new_ver|;

		$LINK->Save();

		$log.=" R1: $LINK->{WorkingDirectory}\n";
		$LINK->Close();
	}
	else{
		die " $mqx_new_ver not match $regex_old_mqx_ver !\n";

	}

#	$LINK->Load($LINK_path) or die "$LINK_path not found!\n";
#	say "print $LINK_path ";
#	print_lnk($LINK);
#	$LINK->Close();
	return $log; 
}#---end process_each_lnk($LINK_path, $mqx_new_ver)

sub pushd_mqx_root_proc_all_lnk($){
	my ($mqx_root)=@_;
	chomp($mqx_root);
#say $mqx_root; 

	my $dir=pushd $mqx_root ;
	`chmod 0755 *.lnk `;
	my $mqx_new_ver=basename($dir); # mqx_new_ver came from MQX_ROOT, seperate ; 
#say $mqx_new_ver;
	my $first_flg=0; 
	map{
	my $log_process_each_lnk=process_each_lnk($_,$mqx_new_ver) ;
	say $log_process_each_lnk ;#if ($first_flag == 0);  #process_each_line($LINK_path, $mqx_new_ver)
	$first_flag++;	
	}glob('*.lnk');

} #--- end pushd_mqx_root_proc_all_lnk($mqx_root); 


my $build_dir = getcwd;   #get curdir; 
die " not in DIR build/\n " if ! ($build_dir =~ m/build/ ) ; # if !() , must have ();  

my $mqx_root=get_mqx_root_from_build_dir($build_dir);

pushd_mqx_root_proc_all_lnk($mqx_root); # mqx_root sperate into dirname and basename 


