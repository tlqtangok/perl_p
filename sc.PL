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

	$ENV{TS} = &get_timestamp();

	my $e_l = '0;$t=($low + $hi * 2)/3.0; $l=$t*0.94; $h=$t*0.97; printf "%.2f/%.2f $ENV{TS}\n",$l ,$h; '; 
	$e_l =~ s/\$low/$low/g;
	$e_l =~ s/\$hi/$hi/g;
	eval $e_l;
}

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
	#$time_stamp = "$year$month${date}_${hour}$minute";
	$time_stamp = "$year$month${date}:";

	chomp( $time_stamp );
	return $time_stamp;
}


