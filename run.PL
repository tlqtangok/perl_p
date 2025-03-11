#!perl
use strict;
#use threads;
#use IO::Handle;
use File::Spec;
my $SP = File::Spec->catfile('', ''); 

my $g_rel_debug = $ARGV[0];

&main($g_rel_debug); 

sub main($)
{

    my $rel_debug = shift;

    chdir ("vs_build"); 


    my $cml_rel = q{
    :: cmake .. -DENABLE_THIRDPARTY=ON -DCMAKE_BUILD_TYPE=Release 
    cmake --build . --config release 

    cp D:\jd\t\git\dna-analysis\vs_build\bin\Release\heer-analysis.win32.amd64.node d:\jd\t\git\analysis-ui\resources\win32\heer-analysis.win32.amd64.node
    cp .\bin\Release\unittest.exe .\bin\Debug\unittest.exe 
    cp .\bin\Release\analysisctl.exe .\bin\Debug\analysisctl.exe 

    dir .\bin\Debug\unittest.exe  d:\jd\t\git\analysis-ui\resources\win32\heer-analysis.win32.amd64.node  |grepw 202


    };

    my $cml_debug = q{

    cmake .. -DENABLE_THIRDPARTY=ON -DCMAKE_BUILD_TYPE=Debug 
    cmake --build . --config Debug 

    :: .\bin\Debug\unittest.exe --gtest_filter=CellTest.analysis 2>&1 | grepw PASSED |grepw /C:"1 test" 
    };


    my $ret_code;
    my @fc = (); 

    my $verbose = 1;

    print "- use config : $rel_debug", "\n\n"; 

    if ($rel_debug eq "rel")
    {
        ($ret_code, @fc) = &run_cmd_bat($cml_rel, $verbose);
    }
    elsif ($rel_debug eq "debug")
    {
        ($ret_code, @fc) = &run_cmd_bat($cml_debug, $verbose);
    }
    else
    {
        die "args error! \n"; 
    }


    &post_process($ret_code, @fc);

}

sub post_process($,@)
{
    my ($ret_code, @fc) = @_; 
    if ($ret_code != 0)
    {
        map{print;}@fc;
    }
    else
    {
        #map{print;}@fc;

       map 
        {
            print if m/^202/;
        }(@fc);
        print "build success!\n"; 
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

    my @fc  = ` $cml  2>&1 `; 
    my $ret_code = $? >> 8; 
    my @stderr = grep {m/run \[.*error/;}@fc;
    #print "@stderr","\n";
    return ($ret_code, @fc); 
}


