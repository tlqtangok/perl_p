#!perl
use strict ;

use File::Basename;
use File::Copy;
use Cwd;
use Data::Dumper;



my @filename_EMAIL_LIST = (@ARGV); 
my $email_cmd =qq| mutt -a  _FILE_NAME_  -s 'see attachment: _FILE_NAME_'  -- EMAIL_LIST < ~/email_templ.txt 
|; 


if ( @filename_EMAIL_LIST == 0 || @filename_EMAIL_LIST == 1 ) {
	print "- Usage:\n\tmail2 ~/txt.txt 'linqi\@SNPSCOM'\n"; 
	exit(1); 
}
elsif ( !(-e @filename_EMAIL_LIST[0]) ){
	print "- @filename_EMAIL_LIST[0] not exists!\n"; 
	print "- Usage:\n\tmail2 ~/txt.txt 'linqi\@SNPSCOM'\n"; 
	exit(2);
}

my %hash_email = (
	linqi => "linqi\@SNPSCOM",
	jd => "linqi\@SNPSCOM",
	jidor => "linqi\@SNPSCOM",
	jt => "lyou\@SNPSCOM",
	yl => "yunluz\@SNPSCOM",
	hf => "haifeng\@SNPSCOM",
	ll => "heli\@SNPSCOM",
	ir => "iphiron\@SNPSCOM",
	mt => "miteshsi\@SNPSCOM",
	tm => "thuymain\@SNPSCOM",
);


#print values %hash_email ; 

my $_FILE_NAME_ = @filename_EMAIL_LIST[0]; 
my $email_list  = " "; 

map{
	my $email_ = @filename_EMAIL_LIST[$_] ; 

	if ($email_ !~ m|\@| ){
		if ( @hash_email{$email_}){  $email_ = @hash_email{$email_}; } 
	}

	die "- $email_ don't match email "  if $email_ !~ m|\@|;
	$email_list .= q(').$email_.q('). " ";
} (1..@filename_EMAIL_LIST-1); 


$email_cmd =~ s/EMAIL_LIST/$email_list/; 
$email_cmd =~ s/_FILE_NAME_/$_FILE_NAME_/;
$_FILE_NAME_ = basename($_FILE_NAME_); 
$email_cmd =~ s/_FILE_NAME_/$_FILE_NAME_/;
print $email_cmd ; 


