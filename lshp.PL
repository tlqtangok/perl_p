#!perl
use feature qw(say); 


my $cml = &main(); 

system ( $cml ); 



sub main()
{

	my $NUM = 7; 

	$NUM = $ARGV[0] if @ARGV;

	my $cml = q{ find . -name '*' | perl -e '@a=<>; chomp(@a); @a = sort{-M $a<=>-M $b}@a; @a = grep  ! m/(^\.$)|(^\.\/$)|(^\.\/\.\w)/, @a; print join "\n",@a[0.._NUM_];print "\n";' };

	$cml =~ s/_NUM_/$NUM/g; 

	#say $cml;
	return $cml ;

}
