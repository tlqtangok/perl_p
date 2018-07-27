#!perl
chdir(q(/media));

my @fn = `ls sf_et |head`; 


if (@fn == 0)
{
	#print "- no mount yet\n";
	my $mnt_statement =  "\tcd /media && sudo mount -t vboxsf et sf_et\n";
	print $mnt_statement;
	system( $mnt_statement);
}

else
{
	my 	$u_mnt_statement = "\tcd /media && sudo umount /media/sf_et\n";
	print $u_mnt_statement;
	system( $u_mnt_statement );
}

