#!perl 
use POSIX qw(assert);
use feature qw(say);

&main(); 
### sub list ###
sub main()
{

	if (@ARGV)
	{
		my $e_l = $ARGV[0];
		$_ = $e_l;

		chomp(); 
		last   if (m/^q$|^quit$|^exit$/i);

		$e_l =~ s/%/\/100/g;
		#say $e_l;

		eval "\$_=$e_l;"; 
		say $_;
	}
	else
	{
		while(my $e_l = <>)
		{

			chomp(); 
			s/%/\/100/g;
			last   if (m/^q$|^quit$|^exit$/i);

			$e_l =~ s/%/\/100/g;
			#say $e_l;

			if (0 && $e_l =~ m/^sc \d/)
			{
				my @arr_num = split m/\s+/, $e_l;
				assert(@arr_num == 3); 
				my $low = $arr_num[1]; 
				my $hi = $arr_num[2];

				if ($low > $hi)
				{
					($low, $hi) = ($hi, $low);
				}
				$e_l = '0;$t=($low + $hi * 2)/3.0; $l=$t*0.94; $h=$t*0.97; printf "%.2f ~ %.2f\n",$l ,$h; '; 
				$e_l =~ s/\$low/$low/g;
				$e_l =~ s/\$hi/$hi/g;
			}

			eval "\$_=$e_l;"; 
			say $_;
		}
	}
}
