#!perl

use feature qw(say); 

map
{

    if (-T "$_")
    {


        print "*********************\n";
        print "$_\n";
        print "*********************\n";

        if ($^O eq "MSWin32") 
        {
            print `type $_ 2>nul`;
        }
        else
        {
            print `cat $_`;
        }


        print "\n\n";

    }

}@ARGV;

print "EOF\n\n";
