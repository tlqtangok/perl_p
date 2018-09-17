#!perl 

#my @fn = `cat 1.txt`; 
my @fn = `cat win.txt`; 

for my $e (@fn)
{
	chomp($e); 
	if (-T $e && $e !~ m|\.git[\/\\]| 
		&& $e !~ m/CMakeFiles[\/\\]/ 
		&& $e !~ m/\.cmake/
		&& $e !~ m/\.txt$/ 
		&& $e !~ m/\.a$/ 
		&& $e !~ m/\.gitignore$/ 

	)
	{
		print ($e, "\n"); 	
	}

}


