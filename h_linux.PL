#!/usr/bin/perl

my @fc = <>;


&main(\@fc);



sub main(\@)
{
	my $ref_fc = shift;
	if (not defined $ref_fc)
	{
		die "Usage: $0 <file>\n";
	}

	my @fc = @$ref_fc;

	my %hash_cml = (); 

	my @result = ();

	for (reverse @fc)
	{

		chomp;

		my $eline = $_;
		my @f = split m/\s+/;
		my $idx = $f[1];  # First element is the number

		my $cml = $eline;
		$cml =~ s/^\s*\d+\s+//;  # Remove leading number and spaces
		$cml =~ s/\s+$//;  # Remove trailing spaces

		#print "___".$cml."___\n";  # Debugging line



		if ($cml =~ m/^h\s+/ 
			|| $cml eq "lsh" 
			|| $cml eq "ls" 
			|| $cml eq "h" 
			|| $cml eq "pwd" 
			|| $cml eq "cd")
		{
			next;  # Skip commands that start with h, ls, cd, or pwd
		}

		# if hash_cml{$cml} is defined, then we have a duplicate

		if (defined $hash_cml{$cml})
		{
			next;
		}

		push @result, "$idx $cml\n";

		$hash_cml{$cml} = 1;  # Mark this cml as seen

	}

	for(reverse @result)
	{
		print $_;
	}

}
