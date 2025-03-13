#!/usr/bin/perl
use strict;
use warnings;
use Cwd qw(abs_path getcwd);
use File::Spec qw(catfile);

# Print information header
# print "Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): 2025-03-13 12:01:21\n";
# print "Current User's Login: tlqtangok\n";

&main();


# s0: echo d:\jd\perl_p\win_bin\cbin_log.bat  | perl torel.PL 
# s1: echo d:\jd\perl_p\win_bin\cbin_log.bat  | perl torel.PL  d:\jd
# s2: echo d:\jd\perl_p\win_bin\cbin_log.bat  | perl torel.PL  filelist.txt
# s3: echo d:\jd\perl_p\win_bin\cbin_log.bat  | perl torel.PL  filelist.txt d:\jd\perl_p
# ======
# s0, pipe
# s1, pipe, passdir 
# s2, file
# s3, file, passdir 

sub main()
{
    my $cwd = getcwd();
    my $SP = File::Spec->catfile('', ''); 


    my @argv = grep {m/\w/;}@ARGV;
    my $argc = scalar @argv;

    my $mode = &which_mode(\@argv);

    my $base_dir = defined($cwd) && -d $cwd ? abs_path($cwd) : abs_path(getcwd());
    my ($base_vol) = File::Spec->splitpath($base_dir, 1);

    if ($mode eq "s1" || $mode eq "s3")
    {

        for(@argv)
        {
            if (-d $_)
            {
                $base_dir = $_;
                last; 
            }
        } 

    }

    my @fc = (); 

    if ($mode eq "s0" || $mode eq "s1")
    {
        @fc = <STDIN>;
    }

    if ($mode eq "s2" || $mode eq "s3")
    {

        my $fn_list = "NULL_FN";  
        for(@argv)
        {
            if (-T $_)
            {
                $fn_list = $_;
                last; 
            }
        } 

        open my $FP, "<", $fn_list or die "cannot open $fn_list"; 
        @fc = <$FP>;
        close $FP;
    }



    for my $e_line (@fc)
    {
        chomp ($e_line);

        my $e_rel_fn = File::Spec->abs2rel($e_line, $base_dir);  

        print ".". $SP. $e_rel_fn, "\n"; 

    }

    # print $mode; 

}


sub which_mode(\@)
{
    my $ref_argv = shift;
    my @argv = @{$ref_argv};
    my $mode = "s0"; # Default mode is s0 (pipe mode)

    if (scalar @argv == 0) 
    {
        $mode = "s0";
    } 
    elsif (scalar @argv == 1) 
    {
        my $arg = $argv[0];
        if (-d $arg) 
        {
            $mode = "s1";
        } elsif (-T $arg) 
        {
            $mode = "s2";
        } else 
        {
            $mode = "s0";
        }
    } 
    elsif (scalar @argv == 2) 
    {
        my ($arg1, $arg2) = @argv;
        if (-T $arg1 && -d $arg2) 
        {
            $mode = "s3";
        } else 
        {
            $mode = "s0";
        }
    }

    return $mode;
}
