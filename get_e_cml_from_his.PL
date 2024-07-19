#!perl
use strict ;
#use feature qw(say);
use List::Util qw(first sum max shuffle maxstr);
#use List::MoreUtils qw(uniq);
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
use File::Spec;
my $SP = File::Spec->catfile('', ''); 



### global var ###

my @filter_param_list = qw(); 
my $curdir = cwd();

#my $edit_log = $curdir."/"."bak_script/watch_folder_log.txt";
my $g_fn_list = "";
my %hash_fn_sz = ();
##################

&main(); 

### sub list ###
sub main()
{
    my $os_type = $^O; # MSWin32 | linux | darwin

    my $line_num = 0; 
    if (@ARGV > 0)
    {
        $line_num = $ARGV[0]; 
    }

    my $roi_cml = "";
    my $fn_cmd_exe_history = $ENV{tmp}.$SP."his_cmd.txt";

    open my $FILE_fn_cmd_exe_history, "<", $fn_cmd_exe_history or die; 
    my @fc_cmd_exe_history = <$FILE_fn_cmd_exe_history>;
    close $FILE_fn_cmd_exe_history; 

    if (@fc_cmd_exe_history < $line_num )
    {
        $roi_cml = "";
    }

    if ("$line_num" =~ m/^\d{1,3}$/)
    {
        $roi_cml = $fc_cmd_exe_history[$line_num]; 
    }



    $roi_cml =~ s/^.*?\d //;

    my $fn_roi_cmd = $ENV{tmp}.$SP."e_his_cmd.bat";
    open my $FILE_roi_cmd, ">" , $fn_roi_cmd or die;
    
    my $fc_cml_to_write = q{@echo off}. "\n".
    $roi_cml.
    q{@echo on}; 

    print $FILE_roi_cmd $fc_cml_to_write;

    close $FILE_roi_cmd; 

    #system($roi_cml); 



}

sub get_timestamp()
{

	my $add_zero_if_lt_10 = sub ($num)
	{
		my $num = shift @_;
		if ($num >=10 ){
			return "$num"
		}
		return "0$num";
	};

	my $time_stamp = "NULL";

	my ($second, $minute, $hour, $date, $month, $yearOffset, $dayOfWeek, $dayOfYear, $daylightSavings) = localtime();
	# 17 16 20 24 7 116 3 236 1
	# 2016-08-24 20:16

	my $year = 1900 + $yearOffset;
	$month = $month + 1 ;
	$month = $add_zero_if_lt_10->($month);
	$date = $add_zero_if_lt_10->($date);
	$hour = $add_zero_if_lt_10->($hour);
	$minute = $add_zero_if_lt_10->($minute);
    #$time_stamp = "$year$month${date}";
	$time_stamp = "$year$month${date}_${hour}$minute";

	chomp( $time_stamp );
	return $time_stamp;
}


sub smart_match2($, \@)
{
    my ($tag, $arr_ref) = @_; 

    my $hit_idx = -1; 
    for(my $i=0;$i<@$arr_ref; $i++)
    {
        if ($arr_ref->[$i] eq $tag)
        {
            $hit_idx = $i; 
            last; 
        }
    }

    return $hit_idx;
}




