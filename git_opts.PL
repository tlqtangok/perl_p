#!perl
use strict;
use warnings;
use List::Util qw(first sum max shuffle maxstr);
use File::Spec;
use File::Spec::Functions;
use File::Basename qw(dirname basename);
use Data::Dumper;
use Scalar::Util qw/reftype/;
use File::Copy;
use File::Find;
use Cwd qw(abs_path cwd);
use POSIX qw(strftime);
use threads;
use IO::Handle;
use File::Path;
use File::Path qw(mkpath);
use File::Spec::Functions qw(canonpath);

### global var ###

my @filter_param_list = qw();
my $curdir = cwd();

my $g_fn_list = "";
my @g_arr_fn_list = ();
my %hash_fn_sz = ();
my $g_t0 = "t0";

my $SP = File::Spec->catfile('', '');
my $is_windows = ($^O eq 'MSWin32');
##################

&main();

### sub list ###

sub main
{
    my $os_type = $^O;

    my $opt_mode = "git_save";

    if (@ARGV > 0)
    {
        $opt_mode = $ARGV[0];
    }

    print "opt_mode:\t$opt_mode", "\n";

    if ($opt_mode =~ m/^git_save/i)
    {
        &git_save();
    }

    if ($opt_mode =~ m/^git_reset/i)
    {
        &git_reset();
    }

    if ($opt_mode =~ m/^git_merge/i)
    {
        &git_merge();
    }
}

sub get_git_commit_id
{
    my $commit_id = `git rev-parse HEAD`;
    chomp($commit_id);

    return $commit_id;
}

sub git_save
{
    system("git fetch");

    
    my @origin_master_or_main = `git branch -r`;
    my $master_or_main_0 = $origin_master_or_main[0];

    $master_or_main_0 =~ s/^.*origin.//;
    chomp($master_or_main_0);


    # print("\$master_or_main_0:$master_or_main_0\n");

    
    my $fc_remote_local = `git rev-list --left-right --count origin/$master_or_main_0...$master_or_main_0`;
    chomp($fc_remote_local);

    # print("fc_remote_local: $fc_remote_local\n");

    my @arr_remote_local = split m/\s+/, $fc_remote_local;
    my $cnt_remote_commit = $arr_remote_local[0] || 0;
    my $cnt_local_tocommit = $arr_remote_local[1] || 0;

    if ($cnt_local_tocommit != 0)
    {
        my $cml_reset_local_comit = "git reset HEAD~" . "${cnt_local_tocommit}";
        print $cml_reset_local_comit, "\n";
        system($cml_reset_local_comit);
    }

    my @fc_local_modify = `git status -s`;
    chomp(@fc_local_modify);

    @fc_local_modify = grep {m/^M.? |^ M.? /} @fc_local_modify;

    map
    { 
        my @a = split m/\s+/;
        $_ = $a[-1];
    } @fc_local_modify;
    
    @fc_local_modify = grep {! -d $_} @fc_local_modify;

    my $HEAD_COMMIT_ID = &get_git_commit_id();

    print $HEAD_COMMIT_ID, "\n";

    $HEAD_COMMIT_ID = substr $HEAD_COMMIT_ID, 0, 7;

    my $fn_all_str = join " ", @fc_local_modify;
    my $fn_tgz = "./$g_t0/$HEAD_COMMIT_ID.tgz";

    my @fn_list_edit = &get_edit_filelist($HEAD_COMMIT_ID);

    my $fn_new_commits = "$HEAD_COMMIT_ID".".txt";

    $fn_tgz = File::Spec->abs2rel($fn_tgz, $curdir);

    my $cml_tar_all = "tar czf ".  $fn_tgz .  " $fn_all_str";

    if (! -d "./$g_t0")
    {
        print ( "make_path $g_t0", "\n" );
        mkpath("./$g_t0");
    }

    my $cnt_same = 0;


    my @cml_vimdiff_all = &tag_edit_fn(\@fc_local_modify, \@fn_list_edit);

    my @fn_edit_same = grep{m/^gvim/;}@cml_vimdiff_all;

    if (@fn_list_edit == 0 )
    {
        print ("- no new commits, don't need merge\n");
    }
    else
    {
        if (@fc_local_modify == 0)
        {
            print ("\n","- no local modify","\n");
            print ("now , run:\n");
            print "######","\n";
            print ("git_reset\n");
            print "######","\n";
        }
        elsif  (@fn_edit_same==0)
        {
            print ("\n","- find local modify, but remote depot change other files","\n");
            print ("now , run:\n");
            print "######","\n";
            print ("git pull\n");
            print "######","\n";
        }
        else
        {
            open my $FP, ">" , $fn_new_commits or die;
            print $FP join "\n", @fn_list_edit;
            close $FP;

            $cml_tar_all .= " $fn_new_commits";

            print $cml_tar_all,"\n";
            system($cml_tar_all);

            unlink($fn_new_commits) or warn "Could not delete $fn_new_commits: $!";

            print "\n";
            print ("now , run:\n");
            print "######","\n";
            print ("git_reset\n");
            print ("git_merge\n");
            print "######","\n";
        }
    }

    return $cml_tar_all;
}

sub git_reset
{
    my $verbose = 0;

    my @origin_master_or_main = `git branch -r`;
    my $origin_master_or_main_0 = $origin_master_or_main[0];

    $origin_master_or_main_0 =~ s/^.*origin/origin/;
    chomp($origin_master_or_main_0);

    my @reset_cmds = 
    (
        "git fetch --all",
        "git reset --hard $origin_master_or_main_0",
        "git pull"
    );

    for my $cmd (@reset_cmds)
    {
        print "Running: $cmd\n";
        system($cmd);
    }

    my $fn_submodule = ".gitmodules";
    if (! -e $fn_submodule)
    {
        return;
    }

    my @subm_list = `git config --file .gitmodules --get-regexp path`;
    chomp(@subm_list);

    if (@subm_list > 0)
    {
        for my $e_subm (@subm_list)
        {
            $e_subm =~ s/^.*\.path\s+//;

            my $CURDIR = cwd();

            chdir($e_subm) or die "chdir $e_subm error\n";

            print("\nsubmodule:\n cd $e_subm\n");

            my $current_branch = `git branch --show-current`;
            chomp($current_branch);

            my @main_or_master = `git ls-remote --symref origin HEAD`;
            my $main_or_master_0 = $main_or_master[0];
            chomp($main_or_master_0);

            if ($current_branch !~ m/^master$/ && $current_branch !~ m/^main$/)
            {
                if ($main_or_master_0 =~ m/main/)
                {
                    system("git checkout main");
                }
                else
                {
                    system("git checkout master");
                }
            }

            print(" run git reset\n");

            for my $cmd (@reset_cmds)
            {
                print "Running: $cmd\n";
                system($cmd);
            }

            chdir($CURDIR) or die "chdir $CURDIR error\n";
        }
    }
}

sub git_merge
{
    my @fc;
    find
    (
        sub
        {
            if ( -f $_ && $_ =~ /\.tgz$/ )
            {
                push @fc, $File::Find::name;
            }
        },
        $g_t0
    );

    @fc = sort { -M $a <=> -M $b } @fc;
    chomp(@fc);

    if (@fc == 0)
    {
        die "No tgz files found in $g_t0\n";
    }

    my $fn_latest = $fc[0];
    my $fn_tgz = File::Spec->abs2rel($fn_latest, $curdir);
    my $cml_tar_all = "tar xzf " . $fn_tgz . " -C $g_t0 2>&1";

    print $cml_tar_all, "\n\n";

    my @fn_local = qx($cml_tar_all);

    map
    { 
        my @a=split m/\s+/;
        $_=$a[-1];
    } @fn_local;

    my $commit_id = basename($fn_tgz);
    $commit_id =~ s/\.tgz//;

    @fn_local = grep {! m/$commit_id/} @fn_local;

    map
    {
        my $fn_git_fetch = $_;
        chomp($fn_git_fetch);
        my $fn = "./$g_t0/$fn_git_fetch";
        $fn = File::Spec->abs2rel($fn, $curdir);
        push @g_arr_fn_list, "${fn}___${fn_git_fetch}";

    } @fn_local;

    my @fn_list_edit = &get_edit_filelist($commit_id);

    my @cml_vimdiff_all = &tag_edit_fn(\@g_arr_fn_list, \@fn_list_edit);

    print ("now , run cml to merge:\n");
    print "######","\n";
    map{print; print "\n";}@cml_vimdiff_all;
    print "######","\n";

    return @cml_vimdiff_all;
}

sub tag_edit_fn
{
    my ($ref_g_arr_fn_list, $ref_fn_list_edit) = @_;

    my @g_arr_fn_list = @$ref_g_arr_fn_list;
    my @fn_list_edit = @$ref_fn_list_edit;

    my @cml_all = ();
    for(my $i=0; $i<@g_arr_fn_list; $i++)
    {
        my $fn_diff = $g_arr_fn_list[$i];
        chomp($fn_diff);

        my @arr_ = split m/___/, $fn_diff;
        $fn_diff = $arr_[-1];

        chomp($fn_diff);

        $fn_diff = &unify_sep($fn_diff, $SP);

        my $flag_edit = 0;
        for(my $j=0; $j<@fn_list_edit; $j++)
        {
            my $fn_edit = $fn_list_edit[$j];
            chomp($fn_edit);
            $fn_edit = &unify_sep($fn_edit, $SP);

            if ($fn_edit eq $fn_diff)
            {
                $flag_edit = 1;
            }
        }

        my $e_cml = "gvim -d .$SP$g_t0$SP$fn_diff $fn_diff";
        if ($flag_edit==1)
        {
        }
        else
        {
            $e_cml =~ s/^gvim /GVIM /;
        }

        push @cml_all, $e_cml;
    }

    return @cml_all;
}

sub save_g_fn_list
{
    my $fn = $File::Find::name;
    if( &if_fn_match_pattern($fn, \@filter_param_list))
    {
        $fn = File::Spec->abs2rel($fn, $curdir);

        my $fn_git_fetch = $fn;
        $fn_git_fetch =~ s|${g_t0}${SP}${SP}||;
        $g_fn_list .= "gvim -d $fn $fn_git_fetch\n";

        push @g_arr_fn_list, "$fn ___ $fn_git_fetch";
    }
}

sub if_fn_match_pattern
{
    my $fn = shift;
    my $ref_filter_param_list = shift;
    my $ret_code = 0;

    my $flag_need_filter = 0;
    if (@$ref_filter_param_list == 0)
    {
        $flag_need_filter = 1;
    }

    if ($flag_need_filter)
    {
        if(-f $fn && $fn !~ m/\.git/ && $fn !~ m/\.swp$/ && $fn !~ m/\.zip$/)
        {
            $ret_code = 1;
        }
    }
    else
    {
        for my $e_filter (@$ref_filter_param_list)
        {
            if ($fn =~ m/\.$e_filter$/)
            {
                if(-f $fn && $fn !~ m/\.git/ )
                {
                    $ret_code = 1;
                }

                last;
            }
        }
    }

    return $ret_code;
}

sub get_simple_cksum
{
    my ($file) = @_;
    open(my $fh, '<', $file) or die "Cannot open file $file: $!";
    binmode $fh;
    my $sum = 0;
    while (my $bytes = <$fh>)
    {
        $sum += length($bytes);
    }
    close($fh);
    return $sum;
}

sub get_edit_filelist
{
    my $commit_id = shift;

    my $fn_changelist_file = "$g_t0". $SP. "$commit_id.txt";

    if (-e $fn_changelist_file)
    {
        my @fc = &read_file_content($fn_changelist_file);
        chomp(@fc);

        if (@fc==0)
        {
            return ();
        }
        else
        {
            return @fc;
        }
    }

    system("git fetch");

    my @fc = `git log --all --name-status 2>&1`;

    my $current_commit_id = &get_git_commit_id();

    my $cnt = 0;
    for (@fc)
    {
        if (m/$current_commit_id/)
        {
            last;
        }
        $cnt++;
    }

    my @fc_roi = @fc[0..$cnt];
    @fc_roi = grep {m/^M\s+/;} @fc_roi;

    map 
    { 
        my @status_fn = split m/M\s+/, $_;
        $_ = $status_fn[1];
    } @fc_roi;

    map
    { 
        $_ = File::Spec->abs2rel($_, $curdir);
    } @fc_roi;

    my %hash_fn = ();
    for(@fc_roi)
    {
        $hash_fn{$_}++;
    }

    my @fn_list = keys %hash_fn;

    return @fn_list;
}

sub read_file_content
{
    my $filename = shift;
    open my $fh, '<', $filename or die "Cannot open $filename: $!";
    my @lines = <$fh>;
    close $fh;
    return @lines;
}

sub post_process
{
    my ($ret_code, @fc) = @_;
    if ($ret_code != 0)
    {
        map{print;} @fc;
    }
    else
    {
        print "run success!\n";
    }
}

sub get_temp_dir
{
    if ($is_windows)
    {
        return $ENV{TEMP} || $ENV{TMP} || 'C:\\temp';
    }
    else
    {
        return $ENV{TMPDIR} || '/tmp';
    }
}

sub run_cmd_bat
{
    my $bat_stat = shift;
    my $verbose = 1;
    if (@_ > 0)
    {
        $verbose = shift;
    }

    $| = 1;

    if ($is_windows)
    {
        my $fn_bat = &cml_to_file($bat_stat, $verbose);
        select(undef, undef, undef, 0.15);
        my ($ret_code, @fc) = &run_cmd("call $fn_bat");

        my $temp_dir = &get_temp_dir();
        my $fn_log = File::Spec->catfile($temp_dir, "run_e_cml.log");

        open my $FILE_log, ">", $fn_log or die;
        print $FILE_log join "", @fc;
        close $FILE_log;

        return ($ret_code, @fc);
    }
    else
    {
        my $fn_sh = &cml_to_shell($bat_stat, $verbose);
        select(undef, undef, undef, 0.15);
        my ($ret_code, @fc) = &run_cmd("bash $fn_sh");

        my $temp_dir = &get_temp_dir();
        my $fn_log = File::Spec->catfile($temp_dir, "run_e_cml.log");

        open my $FILE_log, ">", $fn_log or die;
        print $FILE_log join "", @fc;
        close $FILE_log;

        return ($ret_code, @fc);
    }
}

sub get_stderr
{
    my @fc = @_;
    return grep{m/run \[.*error/;} @fc;
}

sub cml_to_file
{
    my $cml = shift;
    my $verbose = shift;

    my $temp_dir = &get_temp_dir();
    my $fn_cml = File::Spec->catfile($temp_dir, "run_e_cml.bat");
    $| = 1;

    chomp($cml);

    my @fc_cml = split m/\n/, $cml;

    map 
    {
        my $org = $_;
        chomp;
        s/^\s*?(\w)/$1/g;
        s/\s+$//g;

        if ($_ =~ m/^::/)
        {
            $_ = "\n";
            return;
        }

        my $echo_error_content = "run [ $_ ] error !";
        if ($_ =~ m/[\&\>\"\|]/)
        {
            $echo_error_content = q{"}.$echo_error_content.q{"};
        }

        if (length($_)>0)
        {
            my $add_if_stat = qq{if %ERRORLEVEL% NEQ 0 (echo $echo_error_content && exit /b 1)}. "\n";
            if ($_=~m/;$/)
            {
                $add_if_stat = "";
            }
            my $cml_all = $_ . "\n". $add_if_stat;
            $_ = $cml_all;
        }
        else
        {
            $_ = "\n";
        }

    } @fc_cml;

    open my $FILE_cml , ">" , "$fn_cml" or die "open $fn_cml\n";
    if ($verbose == 0)
    {
        print $FILE_cml '@echo off'."\n";
    }

    print $FILE_cml join "", @fc_cml;

    if ($verbose == 0)
    {
        print $FILE_cml '@echo on'."\n";
    }
    close $FILE_cml;

    return $fn_cml;
}

sub cml_to_shell
{
    my $cml = shift;
    my $verbose = shift;

    my $temp_dir = &get_temp_dir();
    my $fn_sh = File::Spec->catfile($temp_dir, "run_e_cml.sh");
    $| = 1;

    chomp($cml);

    my @fc_cml = split m/\n/, $cml;

    map 
    {
        my $org = $_;
        chomp;
        s/^\s*?(\w)/$1/g;
        s/\s+$//g;

        if ($_ =~ m/^::/)
        {
            $_ =~ s/^::/# /;
            $_ .= "\n";
            return;
        }

        if (length($_)>0)
        {
            my $add_if_stat = qq{if [ \$? -ne 0 ]; then echo "run [ $_ ] error !"; exit 1; fi}. "\n";
            if ($_=~m/;$/)
            {
                $add_if_stat = "";
            }
            my $cml_all = $_ . "\n". $add_if_stat;
            $_ = $cml_all;
        }
        else
        {
            $_ = "\n";
        }

    } @fc_cml;

    open my $FILE_sh , ">" , "$fn_sh" or die "open $fn_sh\n";

    print $FILE_sh "#!/bin/bash\n";

    if ($verbose == 0)
    {
        print $FILE_sh "set +x\n";
    }
    else
    {
        print $FILE_sh "set -x\n";
    }

    print $FILE_sh join "", @fc_cml;

    close $FILE_sh;

    chmod 0755, $fn_sh;

    return $fn_sh;
}

sub run_cmd
{
    my $cml = shift;

    my @fc = `$cml 2>&1`;
    my $ret_code = $? >> 8;
    my @stderr = grep {m/run \[.*error/;} @fc;

    return ($ret_code, @fc);
}

sub unify_sep
{
    my ($path, $SP) = @_;
    return canonpath($path);
}

1;



