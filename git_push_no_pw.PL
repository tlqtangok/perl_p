#!perl

&main(); 

### sub list ###
sub main()
{
	my @fn = (@ARGV); 

	die "- please input args, filename or commit_msg\n" if @fn == 0; 

	my $commit_msg = ""; 
	for my $e_fn (@fn)
	{
		if (-e $e_fn)
		{
			system(qq{
				git add $e_fn
				}); 
		}
		else
		{
			$commit_msg = $e_fn; 
		}
	}

	my $commit = join " " , @fn ;
	$commit .= " at " . &get_timestamp(); 	

	$commit_msg = qq{add or edit: $commit} if $commit_msg eq "";


	system(qq{
		git commit  -m'$commit_msg' 
		git push
		}); 
}


sub get_timestamp(){

	my $add_zero_if_lt_10 = sub ($num)
	{
		my $num = shift @_;
		if ($num >=10 ){
			return "$num"
		}
		return "0$num";
	};

	# perl get time, for second:
	#perl -e 'print $^T '
	#1477400615
	my $if_windows = "True";

	my $time_stamp = `date "+%Y%m%d_%H%M"` if $if_windows ne "True";

	if ( $if_windows eq "True"){
		my ($second, $minute, $hour, $date, $month, $yearOffset, $dayOfWeek, $dayOfYear, $daylightSavings) = localtime();
		# 17 16 20 24 7 116 3 236 1
		# 2016-08-24 20:16

		my $year = 1900 + $yearOffset;
		$month = $month + 1 ;
		$month = $add_zero_if_lt_10->($month);
		$date = $add_zero_if_lt_10->($date);
		$hour = $add_zero_if_lt_10->($hour);
		$minute = $add_zero_if_lt_10->($minute);
		$time_stamp = "$year$month${date}_${hour}$minute";
	}

	chomp( $time_stamp );
	return $time_stamp;
}


