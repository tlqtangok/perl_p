#!perl

# create by jidor tang at 2019.03.28 
# to record the ip address at each reboot time
# record to algo , to redis : 00_0a_f7_bc_c3_26.tell_my_ip.txt
# deps : $t, $perl_p/tfr and tell_my_ip.PL 

# usage:
#   perl tell_my_ip.PL    # upload my ip 
#   perl tell_my_ip.PL 18_66_da_f4_ca_e0  # query an ip 
#   perl tell_my_ip.PL all # query all ip based on hwaddr

use strict;
use feature qw(say); 



&main(); 

sub main()
{

	use FindBin qw($Bin);
	my $SCRIPT_ROOT = $Bin;
	chdir($SCRIPT_ROOT);

	my $perl_p = $SCRIPT_ROOT; 

	$ENV{perl_p} = $perl_p;
	$ENV{t} = $perl_p;  

	if (@ARGV == 0)
	{

		my $hwaddr = &get_HWaddr(); 
		my $tor_file_full = $ENV{t}. "/" . $hwaddr. ".tell_my_ip.txt"; 

		qx{ ifconfig > $tor_file_full 2>&1 }; 

		my $return_of_tor = qx{ $perl_p/tfr tor $tor_file_full };
		say $return_of_tor; 

	}
	else
	{
		if ($ARGV[0] =~ m/_/)
		{
			my @arr_hwaddr = split /_/, $ARGV[0]; 

			die "- please input HWaddr, such as 18_66_da_f4_ca_e0\n" if @arr_hwaddr !=  6; 
			my $HWaddr = $ARGV[0]; 
			if (-e "${HWaddr}.tell_my_ip.txt")
			{
				unlink( "${HWaddr}.tell_my_ip.txt" );
			}

			my $DIS_PW_PREFIX_ = "JD_DISABLE_PW_"; 
			my $ts_day = &get_timestamp();  
			my $ts_jd_xx_fn =  qx{ echo ${DIS_PW_PREFIX_}${ts_day} | $perl_p/tfr show_visitor  |grep $HWaddr| grep tor  |tail -n 1 };
			#say $ts_jd_xx_fn;  # 20190328_2056	tor	116.6.99.221	995	0	jd_195	00_0a_f7_bc_c3_26.tell_my_ip.txt

			if ($ts_jd_xx_fn !~ m/jd_\d{1,3}/)
			{
				say "- your ip has not record yet\n"; 	
				exit(-1);
			}

			else
			{
				chomp($ts_jd_xx_fn); 
				my @arr_record_of_visitor = split m/\t/, $ts_jd_xx_fn; 
				die if $arr_record_of_visitor[1] ne "tor"; 

				my $jd_xx = $arr_record_of_visitor[-2]; 

				die if $jd_xx !~ m/jd_\d{1,3}/; 

				say "- the last time record your ip was $arr_record_of_visitor[0]\n-------------\n";  

				my @cmd_fr_ans = qx{ $perl_p/tfr fr $jd_xx };

				my $fn_ip = $cmd_fr_ans[-1]; 
				my @arr_ip_fn = split m/\s+/, $fn_ip;
				$fn_ip = @arr_ip_fn[-1]; 
				my $new_ip = `cat $fn_ip`; 
				say $new_ip; 
			}
		}
		else # all
		{

			die "- please input HWaddr, such as 18_66_da_f4_ca_e0\n" if $ARGV[0] ne "all"; 
			my $all = $ARGV[0]; 

			my $DIS_PW_PREFIX_ = "JD_DISABLE_PW_"; 
			my $ts_day = &get_timestamp();  
			my $ts_jd_xx_fn =  qx{ echo ${DIS_PW_PREFIX_}${ts_day} | $perl_p/tfr show_visitor  |grep tell_my_ip.txt | grep tor |tail -n 255};
			say $ts_jd_xx_fn; 
		}
	}


	# /bin/su - phoenix -c "sh /home/phoenix/t/perl_p/tell_my_ip.PL &"
}




### sub list 

sub get_timestamp()
{

	my $add_zero_if_lt_10 = sub ($num)
	{
		my $num = shift @_;
		if ($num >=10 ){
			return "$num"
		}
		return "0$num";
	};

	my $time_stamp = "NULL";

	my ($second, $minute, $hour, $date, $month, $yearOffset, $dayOfWeek, $dayOfYear, $daylightSavings) = localtime();
	# 17 16 20 24 7 116 3 236 1
	# 2016-08-24 20:16

	my $year = 1900 + $yearOffset;
	$month = $month + 1 ;
	$month = $add_zero_if_lt_10->($month);
	$date = $add_zero_if_lt_10->($date);
	$hour = $add_zero_if_lt_10->($hour);
	$minute = $add_zero_if_lt_10->($minute);
	$time_stamp = "$year$month${date}";
	#$time_stamp = "$year$month${date}_${hour}$minute";

	chomp( $time_stamp );
	return $time_stamp;
}


sub get_HWaddr()
{
	my @cmd_out_ifconfig = qx( ifconfig ); 
	chomp(@cmd_out_ifconfig); 

	my $cmd_out_hostname = qx( hostname -I ); 

	my @arr_ips = split m/\s+/, $cmd_out_hostname; 
	my $true_ip = $arr_ips[0]; 
	my $i = 0; 

	my $ip_line = 0; 
	for( $i=0; $i<@cmd_out_ifconfig; $i++)
	{
		if ( index("$cmd_out_ifconfig[$i]",  "$true_ip") > 0 )
		{
			$ip_line = $i; 
		}
	}
	my $hwaddr_line = 0; 
	my $hwaddr = "NULL"; 

	for( $i = $ip_line - 1; $i<$ip_line + 6; $i++ )
	{
		if ( $cmd_out_ifconfig[$i] =~ m/HWaddr/ )
		{
			$hwaddr_line = $i; 
			my @arr = split m/\s+/, $cmd_out_ifconfig[$hwaddr_line];
			$hwaddr = $arr[-1]; 
			last;
		}
		if ($cmd_out_ifconfig[$i] =~ m/ether/)
		{
			$hwaddr_line = $i; 
			$cmd_out_ifconfig[$hwaddr_line] =~ s/^.*ether\s+//;
			my @arr = split m/\s+/, $cmd_out_ifconfig[$hwaddr_line];
			$hwaddr = $arr[0]; 
			last;
		}
	}

	die "hwadd should has ':' \n" if $hwaddr !~ m/\:/;
	$hwaddr =~ s/\:/_/g;
	chomp($hwaddr);
	return $hwaddr; 
}

