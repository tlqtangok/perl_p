#!/usr/bin/perl
use strict;
use warnings;
use File::Copy qw(copy);

# Get the number of input parameters

my @argv = ();

for(@ARGV)
{
    push @argv, $_ if $_ ne "";
}

my $pcnt = scalar @argv;

@ARGV = (@argv);

# Exit with code -1 if parameter count is not 2 or 3
if ($pcnt != 2 && $pcnt != 3) {
    print "Error: Script requires exactly 2 or 3 parameters.\n";
    print "Usage:\n";
    print "  $0 LOCAL_FILE REMOTE_FILE              - Compare two files\n";
    print "  $0 LOCAL_FILE REMOTE_FILE RESULT_FILE  - Compare and save to result\n";
    exit -1;
}

if ($pcnt == 2) {
    my ($local_file, $remote_file) = @ARGV;

    print "LOCAL: $local_file\n";
    print "REMOTE: $remote_file\n";

    # Start gvimdiff
    system("gvim -f -d \"$local_file\" \"$remote_file\"");
    exit $? >> 8;
}

if ($pcnt == 3) {
    my ($local_file, $remote_file, $result_file) = @ARGV;

    print "LOCAL: $local_file\n";
    print "REMOTE: $remote_file\n";
    print "TO_SAVE: $result_file\n";

    print "cp -p \"$remote_file\" \"$result_file\"\n";
    if (!copy($remote_file, $result_file)) {
        print "Error: Failed to copy \"$remote_file\" to \"$result_file\"\n";
        exit 1;
    }

    # Start gvimdiff
    system("gvim -f -d \"$local_file\" \"$result_file\"");
    exit $? >> 8;
}
