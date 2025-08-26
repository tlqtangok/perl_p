#!perl
use strict ;
#use feature qw(say);
use List::Util qw(first sum max shuffle maxstr);
#use List::MoreUtils qw(uniq);
#
use File::Spec;
use File::Spec::Functions;
use File::Basename qw(dirname basename);
use Data::Dumper;
use Scalar::Util qw/reftype/;
use File::Copy;
use File::Find;
use Cwd qw(abs_path cwd);
use POSIX qw(assert);
use strict;
use threads;
use IO::Handle;
use File::Path;
use File::Path qw(mkpath);
use File::Spec::Functions qw( canonpath );  



### global var ###
my $curdir = cwd();
# my $SP = File::Spec->catfile('', ''); 
##################

&main(); 

### sub list ###
# main_
sub main()
{

    my $OS = $^O;

    my $fn_git_submodule = ".gitmodules";

    my $cml_git_status_s = "git status -s"; 

    my @fc_modify = `$cml_git_status_s`;

    @fc_modify = grep{ m/^\s+M\s+/; }@fc_modify;


    print join "",@fc_modify;

    map{
    chomp; 
    my @arr = split m/M\s+/, $_;
    $_ = $arr[-1];
    }@fc_modify;

    if (-e $fn_git_submodule)
    {
        @fc_modify = grep {-f $_;}@fc_modify;
    }

    my $cml_git_add ="\n".  " git add " . join " ",@fc_modify;
    print $cml_git_add, "\n";

    # gitsubmodule 
    if (-e $fn_git_submodule)
    {

        my @subm_list = `git config --file .gitmodules --get-regexp path`;
        chomp(@subm_list);

        for my $e_subm (@subm_list)
        {
            $e_subm =~ s/^.*\.path\s+//;

            my $CURDIR = cwd();

            chdir($e_subm) or die "chdir $e_subm error\n";

            my @fc = `$cml_git_status_s`;
            @fc = grep{ m/^\s+M\s+/; }@fc;

            if (@fc>0)
            {
                # print("\ncd $e_subm\n");
                my $cml_cd = "pwd"; 
                if ($OS =~ m/Win32/)
                {
                    $cml_cd = "pwd";
                }

                my $cd_win = `$cml_cd`;

                print "\nsubmodule:\n", "cd $cd_win";
                system("gs");
            }

            chdir($CURDIR) or die "chdir $CURDIR error\n";
        }

    }
}

