#!perl
use strict;
#use threads;
#use IO::Handle;
use File::Spec;
use Cwd qw(abs_path cwd);
use File::Basename qw(dirname);
use File::Spec::Functions qw(rel2abs);

my $SP = File::Spec->catfile('', ''); 


my $g_arg0 = $ARGV[0];

chdir( dirname( rel2abs( __FILE__ ) ) );
&main($g_arg0); 

sub main($)
{


    # print (cwd());
    
    if ( -d "c:/jd" || -d "d:/jd" )
    {
        die "error, -d d:/jd or -d c:/jd \n"; 
    }


    my $cml_stat = q{
        :: rm -rf c:/jd d:/jd

        copy jd.tgz d:\
        mklink /j c:\jd d:\jd ;
        pushd d:\
        tar xzf jd.tgz
        del jd.tgz
        REM NOW RUN : COPY d:\jd\perl_p\tt.bat c:\windows
        start explorer.exe /select, d:\jd\perl_p\tt.bat
        
    };


    my $verbose = 1;
    my ($retcode, @fc) = &run_cmd_bat($cml_stat, $verbose);

    if ($retcode != 0)
    {
        map{print;}@fc;
    }

    print q{COPY d:\jd\perl_p\tt.bat c:\windows}, "\n";

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
        #map{print;}@fc;
        @fc = grep {m/202.*(exe|node)/;}@fc;
        print join "", @fc; 
        print "run testcase success!\n"; 
    }

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


