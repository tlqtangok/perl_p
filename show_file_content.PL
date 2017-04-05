#!perl

use feature qw(say); 

map{
if(-T "$_") {
say "*********************";
say ;
say "*********************";
say `cat $_`;
	
	
	}


}@ARGV;

say "EOF\n";
