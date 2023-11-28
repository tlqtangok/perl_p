#perl
use File::Copy;
use feature qw(say); 
use File::Basename qw(dirname basename);
use Data::Dumper; 


&main();

### sub list ###
sub main()
{

	my $h_standard_id_mnt = 
	{
		'fc94b2b5-07fd-4e95-817c-a10766259b3b' => '/mnt/dev_sdb1',
		'eff6e7ad-4d0f-4156-9799-35f8474b6233' => '/mnt/dev_sdb2',
		'85be5f19-fee0-4395-8150-b092b457a7f7' => '/mnt/dev_sdb3',
	};

	my $h_id_to_dev = {}; 


	my $ff = "perl $ENV{perl_p}/ff.PL ";
	my @mnt_list = `lsblk -f |$ff 2 0 -1|grep mnt|sort `;

	my @arr_mnt_cml_res = (); 

	for (@mnt_list)
	{

		chomp;
		my @e_list = split m/ /, $_;
		$e_list[1] = "/dev/". $e_list[1];

		#say join ":",($e_list[0], $e_list[1], $e_list[2]); 

		$h_id_to_dev->{$e_list[0]} =  $e_list[1];
	}


	for (keys %{ $h_standard_id_mnt })
	{
		my $e_k = $_;
		my $e_cml = "sudo mount $h_id_to_dev->{$e_k} $h_standard_id_mnt->{$e_k}";
		push @arr_mnt_cml_res , $e_cml;
	}

	#say Dumper($h_id_to_dev);
	#say Dumper($h_standard_id_mnt);

	#say Dumper(\@arr_mnt_cml_res); 
	
	say join "\n", @arr_mnt_cml_res;
}




