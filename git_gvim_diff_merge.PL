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
    my ($remote_file, $local_file) = @ARGV;

    print "P0: $remote_file\n";
    print "P1: $local_file\n";
    print "\n";


    my $local_win_path = $local_file;
    $local_win_path =~ s|\\|\/|g;  # to unix style path

    # print $local_win_path . "\n";
    open my $fh_local, '<', $local_file or die "Could not open file '$local_file' $!";
    my @fc_local = <$fh_local>;
    close $fh_local;

    # print "line 1: " . $fc_local[0] . "\n";
    # print "Local file content:\n";
    # print @fc_local;


    if (@fc_local == 1 && $fc_local[0] =~ /^Subproject commit /)  # Check for submodule commit line
    {
        print "it is submodule, do nothing\n";
        exit 0;
    }

    # Start gvimdiff
    system("gvim -f -d \"$local_file\" \"$remote_file\"");
    exit $? >> 8;
}

if ($pcnt == 3) {
    my ($local_file, $remote_file, $result_file) = @ARGV;

    print "LOCAL: $local_file\n";
    print "REMOTE: $remote_file\n";
    print "TO_SAVE: $result_file\n";
    print "\n";

    print "cp -p \"$remote_file\" \"$result_file\"\n";
    if (!copy($remote_file, $result_file)) {
        print "Error: Failed to copy \"$remote_file\" to \"$result_file\"\n";
        exit 1;
    }

    # Start gvimdiff
    system("gvim -f -d \"$local_file\" \"$result_file\"");
    exit $? >> 8;
}
