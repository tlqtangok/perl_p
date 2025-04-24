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

my @filter_param_list = qw(); 
my $curdir = cwd();

#my $edit_log = $curdir."/"."bak_script/watch_folder_log.txt";
my $g_fn_list = "";

my @g_arr_fn_list = ();

my %hash_fn_sz = ();
my $g_t0 = "t0";

my $SP = File::Spec->catfile('', ''); 
##################

&main(); 

### sub list ###
# main_
sub main()
{
    my $os_type = $^O; # MSWin32 | linux | darwin


   my $opt_mode = "git_save";  

    # my $num_topn = 11; 
    if (@ARGV > 0)
    {
        $opt_mode = $ARGV[0];     
    }


    # print $opt_mode; 

    #cat fn_all.txt |perl -e '@a=<>;chomp(@a);@aa=sort {-s $b <=> -s $a} @a;print "@aa[0..10]";'

    # print "$curdir\n";  # d:/jd/t/test


    # git_save
    # git_reset
    # git_merge

    # my $cml_tar_all = &git_save();


    if ($opt_mode eq "git_save")
    {
        &git_save(); 
    }


    if ($opt_mode eq "git_reset")
    {
       &git_reset(); 
    }


    if ($opt_mode eq "git_merge")
    {
        &git_merge();
    }

}

sub get_git_commit_id()
{
    my $HEAD_COMMIT_ID = `git rev-parse HEAD`;
    chomp($HEAD_COMMIT_ID);
    return $HEAD_COMMIT_ID;
}

sub git_save()
{
    system("git fetch");

    # count remote and local commit number "5    3" 
    my $fc_remote_local = `git rev-list --left-right --count origin/master...master`;
    chomp($fc_remote_local);

    my @arr_remote_local = split m/\s+/, $fc_remote_local;
    my $cnt_remote_commit = $arr_remote_local[0];
    my $cnt_local_tocommit = $arr_remote_local[1];

    if ($cnt_local_tocommit != 0)
    {
        my $cml_reset_local_comit = "git reset HEAD~" . "${cnt_local_tocommit}"; 
        print $cml_reset_local_comit, "\n"; 
        system($cml_reset_local_comit); 
    }


    my @fc_local_modify = `git status -s`; 
    chomp(@fc_local_modify);

    @fc_local_modify = grep {m/^M.? |^ M.? /} @fc_local_modify;

    # strip ^M 
    map{ my @a = split m/\s+/;    $_ = $a[-1]; }@fc_local_modify; 

    my $HEAD_COMMIT_ID = &get_git_commit_id();

    $HEAD_COMMIT_ID = substr $HEAD_COMMIT_ID, 0, 7;

    my $fn_all_str = join " ", @fc_local_modify;
    my $fn_tgz = "./$g_t0/$HEAD_COMMIT_ID.tgz";

    my @fn_list_edit = &get_edit_filelist($HEAD_COMMIT_ID);


    # print "get_edit_filelist: @fn_list_edit"; 
    # print("\n"); 

    my $fn_new_commits = "$HEAD_COMMIT_ID".".txt";

    

    

    $fn_tgz = File::Spec->abs2rel($fn_tgz, $curdir);  

    
    my $cml_tar_all = "tar cvzf ".  $fn_tgz .  " $fn_all_str";
    # print "cml_tar_all: $cml_tar_all"; 
    # print("\n"); 

    if (! -d "./$g_t0")
    {
        print ( "make_path $g_t0", "\n" );
        mkpath("./$g_t0");
    }


    my $cnt_same = 0;



    # print "fc_local_modify: @fc_local_modify";
    # print("\n"); 
    # print "fn_list_edit:@fn_list_edit"; 
    # print("\n"); 

    my @cml_vimdiff_all = &tag_edit_fn(\@fc_local_modify, \@fn_list_edit);

    # print "cml_vimdiff_all: @cml_vimdiff_all"; 
    # print("\n"); 

   
    my @fn_edit_same = grep{m/^gvim/;}@cml_vimdiff_all;
   


    if (@fn_list_edit == 0 )
    {
        # no summits
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
            # have commits and local modify
            open my $FP, ">" , $fn_new_commits or die;
            print $FP join "\n", @fn_list_edit;
            close $FP; 

            $cml_tar_all .= " $fn_new_commits";

            print $cml_tar_all,"\n";
            system($cml_tar_all); 

            system("del $fn_new_commits");

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

sub git_reset()
{

    my $verbose = 0;
    my $TMP = $ENV{tmp};

    my @origin_master_or_main = `git branch -r`;
    my $origin_master_or_main_0 = $origin_master_or_main[0];

    $origin_master_or_main_0 =~ s/^.*origin/origin/;

    my $cml = qq{

    git fetch --all
    git reset --hard $origin_master_or_main_0
    git pull 

    };


    my ($ret_code, @fc) = &run_cmd_bat($cml, $verbose);

    # &post_process($ret_code, @fc);

}

sub git_merge()
{

    my @fc = `dir /s /b /o-D $g_t0\\*.tgz`;
    chomp(@fc);

    assert(@fc>0);


    my $fn_latest = $fc[0];
    my $fn_tgz = File::Spec->abs2rel($fn_latest, $curdir);  
    my $cml_tar_all = "tar xvzf " . $fn_tgz . " -C $g_t0  2>&1 "; 



    print $cml_tar_all, "\n\n"; 

    my @fn_local = qx($cml_tar_all);  # system


    # print "fn_local: @fn_local"; 
    # print("\n"); 

    map{ my @a=split m/\s+/; $_=$a[-1]; }@fn_local;

    # print "fn_local: @fn_local"; 
    # print("\n"); 
    


    my $HEAD_COMMIT_ID = basename($fn_tgz);
    $HEAD_COMMIT_ID =~ s/\.tgz//;

    @fn_local = grep {! m/$HEAD_COMMIT_ID/} @fn_local;

    # print "fn_local: @fn_local"; 
    # print("\n"); 

    map
    {

        my $fn_git_fetch = $_;
        chomp($fn_git_fetch);
        my $fn = "./$g_t0/$fn_git_fetch"; 
        $fn = File::Spec->abs2rel($fn, $curdir);  
        push @g_arr_fn_list, "${fn}___${fn_git_fetch}"; 

    }@fn_local;

    # print "g_arr_fn_list: @g_arr_fn_list";
    # print("\n"); 


    my @fn_list_edit = &get_edit_filelist($HEAD_COMMIT_ID);


    # print "fn_list_edit:@fn_list_edit";
    # print "\n"; 


    my @cml_vimdiff_all = &tag_edit_fn(\@g_arr_fn_list, \@fn_list_edit);




    print ("now , run cml to merge:\n");
    print "######","\n";
    map{print; print "\n";}@cml_vimdiff_all;
    print "######","\n";

    return @cml_vimdiff_all;

}

sub tag_edit_fn(\@g_arr_fn_list, \@fn_list_edit)
{

    my ($ref_g_arr_fn_list, $ref_fn_list_edit) = @_;

    my @g_arr_fn_list = @$ref_g_arr_fn_list;
    my @fn_list_edit = @$ref_fn_list_edit;

    my @cml_all = ();
    for(my $i=0;$i<@g_arr_fn_list;$i++)
    {
        my $fn_diff = $g_arr_fn_list[$i]; 
        chomp($fn_diff);

        my @arr_ = split m/___/, $fn_diff;
        $fn_diff = $arr_[-1]; 

        chomp($fn_diff);


        $fn_diff = &unify_sep($fn_diff, $SP);
       

        # my @arr_file = split m/ ___ /, $fn_diff;
        # $fn_diff = $arr_file[1];
       

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

sub save_g_fn_list()
{
        my $fn = $File::Find::name;
        if( &if_fn_match_pattern($fn, \@filter_param_list))
        {

            $fn = File::Spec->abs2rel($fn, $curdir);  

            # 47:/
            # 92:\
            my $fn_git_fetch = $fn; 
            $fn_git_fetch =~ s|${g_t0}${SP}${SP}||;
            $g_fn_list .= "gvim -d $fn $fn_git_fetch\n";

            push @g_arr_fn_list, "$fn ___ $fn_git_fetch"; 
        }
}

sub if_fn_match_pattern($fn, $ref_filter_param_list)
{
	my $fn = shift;
	my $ref_filter_param_list = shift; 
	my $ret_code = 0;
	# c cc h hpp cu cuh cpp cxx

	my $flag_need_filter = 0; 	
	if (@$ref_filter_param_list == 0)
	{
		$flag_need_filter = 1; 	
	}

	if ($flag_need_filter)
    {
        if(-f $fn && $fn !~ m/\.git/ && $fn !~ m/\.swp$/
            && $fn !~ m/\.zip$/
        )
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
    binmode $fh;  # Ensure we read in binary mode  
    my $sum = 0;  
    while (my $bytes = <$fh>) {  
        $sum += length($bytes);       # You can change this to a more complex checksum if needed  
    }  
    close($fh);  
    return $sum;  
}  

sub get_edit_filelist($commitid)
{
    # my $tmp = $ENV{tmp};
    my $HEAD_COMMIT_ID = shift;

    my $fn_changelist_file =  "$g_t0". $SP. "$HEAD_COMMIT_ID.txt";


    if (-e $fn_changelist_file)
    {

        my @fc = `type $fn_changelist_file`;
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



    my @fc = `git log --all --name-status 2>&1 `;

    my $HEAD_COMMIT_ID = &get_git_commit_id();

    my $cnt = 0;
    for (@fc)
    {
        if (m/$HEAD_COMMIT_ID/)
        {
            last;
        }
        $cnt++;
    }


    my @fc_roi = @fc[0..$cnt];
    @fc_roi = grep {m/^M\s+/;}@fc_roi;

    map { my @status_fn = split m/M\s+/,$_; $_=$status_fn[1];  }@fc_roi;

    map{ $_ = File::Spec->abs2rel($_, $curdir); }@fc_roi;


    my %hash_fn =();
    for(@fc_roi)
    {
        $hash_fn{$_}++;
    }

    my @fn_list = keys %hash_fn;

    # print @fn_list;
    return @fn_list;
}


sub post_process($,@)
{
    my ($ret_code, @fc) = @_; 
    if ($ret_code != 0)
    { 
        # error
        map{print;}@fc;
    }
    else
    {
        print "run success!\n"; 
    }

}

sub run_cmd_bat($bat_stat,$verbose)
{
    my $bat_stat = shift;
    my $verbose = 1;
    if (@_ > 0)
    {
        $verbose = shift;
    }

    $| = 1;
    my $fn_bat = &cml_to_file($bat_stat, $verbose);
    select(undef,undef,undef,0.15); # sleep 0.15s
    my ($ret_code, @fc) =  &run_cmd("call $fn_bat"); 

  
    

    my $fn_log = $ENV{tmp}.$SP."run_e_cml.log";


    open my $FILE_log, ">", $fn_log or die; 
    print $FILE_log join "",@fc;
    close $FILE_log;

    return ($ret_code, @fc); 
}


sub get_stderr(@)
{
    my @fc = @_;
    return grep{m/run \[.*error/;}@fc;
}

sub cml_to_file($cml,$verbose)
{
    my $cml = shift; 
    my $verbose = shift;

    my $fn_cml = $ENV{tmp}.$SP."run_e_cml.bat";
    $| = 1;

    chomp($cml);

    my @fc_cml = split m/\n/, $cml;

    map {
        my $org = $_;
        chomp;
        s/^\s*?(\w)/\1/g;
        s/\s+$//g;


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

    }@fc_cml;

    open my $FILE_cml , ">" , "$fn_cml" or die "open $fn_cml\n";
    if ($verbose == 0)
    {
        print $FILE_cml '@echo off'."\n";
    }

    print $FILE_cml join "",@fc_cml;

    if ($verbose == 0)
    {
        print $FILE_cml '@echo on'."\n";
    }
    close $FILE_cml;

    return $fn_cml; 
}

sub run_cmd($cml)
{
    my $cml = shift; 

    #my @fc  = ` $cml  2>&1 `; 
    my @fc  = system($cml); 
    my $ret_code = $? >> 8; 
    my @stderr = grep {m/run \[.*error/;}@fc;
    #print "@stderr","\n";
    return ($ret_code, @fc); 
}


sub unify_sep($path, $SP)
{
    my ($path,$SP) = @_;  
    return canonpath($path);
}


   
