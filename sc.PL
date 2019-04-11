#!perl 
use POSIX qw(assert);
use feature qw(say);

&main();


### sub list ###
sub main()
{
	assert(@ARGV == 2); 
	my $low = $ARGV[0]; 
	my $hi = $ARGV[1];

	if ($low > $hi)
	{
		($low, $hi) = ($hi, $low);
	}
	my $e_l = '0;$t=($low + $hi * 2)/3.0; $l=$t*0.94; $h=$t*0.97; printf "%.2f ~ %.2f\n",$l ,$h; '; 
	$e_l =~ s/\$low/$low/g;
	$e_l =~ s/\$hi/$hi/g;
	eval $e_l;
}

