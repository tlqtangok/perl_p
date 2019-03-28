#!perl

# create by jidor tang at 2019.03.28 
# to record the ip address at each reboot time
# record to algo , to redis : 00_0a_f7_bc_c3_26.tell_my_ip.txt
# deps : $t, $perl_p/tfr and tell_my_ip.PL 

&main(); 

sub main()
{
	use FindBin qw($Bin);
	my $SCRIPT_ROOT = $Bin;
	chdir($SCRIPT_ROOT);

	my $perl_p = $SCRIPT_ROOT; 

	$ENV{perl_p} = $perl_p;
	$ENV{t} = $perl_p;  


	my $cmd_get_hwaddr = q{ ifconfig |grep HWaddr|perl $perl_p/ff.PL -1| perl $perl_p/map_.PL 's/:/_/g' }; 

	$cmd_get_hwaddr =~ s/\$perl_p/$perl_p/g; 

	my $fn = qx{ $cmd_get_hwaddr }; 
	chomp($fn); 

	my $tor_file_full = $ENV{t}. "/" . $fn. ".tell_my_ip.txt"; 

	qx{ ifconfig > $tor_file_full 2>&1 }; 

	my $return_of_tor = qx{ $perl_p/tfr tor $tor_file_full };
	print $return_of_tor, "\n"; 

	# /bin/su - phoenix -c "sh /home/phoenix/t/perl_p/tell_my_ip.PL &"
}
