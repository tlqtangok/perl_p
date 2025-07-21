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
my $g_cml_pat = $ARGV[0] || "NULL_PAT";
my $g_roi_num = 5000;
my $g_fn_cmd_exe_history = $ENV{tmp}.$SP."his_cmd.txt";
##################

&main($g_cml_pat, $g_roi_num, $g_fn_cmd_exe_history); 

### sub list ###
sub main($,$,$)
{
    $| = 1;
    my $cml_pat  = shift; 
    my $roi_num  = shift; 
    my $fn_cmd_exe_history = shift;

    my $os_type = $^O; # MSWin32 | linux | darwin



    # read cmd txt , get roi lines,strip the idx
    my @fc_cmd_exe_history = ();

    open my $FILE_fn_cmd_exe_history, "<", $fn_cmd_exe_history;
    if ( not defined $FILE_fn_cmd_exe_history)
    {}
    else
    {
        @fc_cmd_exe_history = <$FILE_fn_cmd_exe_history>;
        close $FILE_fn_cmd_exe_history;
    }

    if (@fc_cmd_exe_history>$roi_num)
    {
        my $sloc = @fc_cmd_exe_history - $roi_num;
        my $eloc = @fc_cmd_exe_history - 1;
        @fc_cmd_exe_history = @fc_cmd_exe_history[$sloc..$eloc];
    }
    map { $_ =~ s/^.*?\d //; }(@fc_cmd_exe_history);
    ######
    

    # read this windows dos cmd history, append to global history
    my @this_history = `doskey /HISTORY`;
    @this_history = grep {!m/^h.?$/} @this_history; 
    @this_history = grep {!m/^! /} @this_history; 

    my $old_last_cml = $fc_cmd_exe_history[-1]; 

    for my $e (@this_history)
    {
        my $new_insert = $e; 
        my $idx_exist = &smart_match2($e, \@fc_cmd_exe_history);
        if ($idx_exist != -1)  # exists
        {
            my $new_insert = splice @fc_cmd_exe_history, $idx_exist,1;
        }

        if ($new_insert !~ m/h.$/)
        {
            push @fc_cmd_exe_history, $new_insert;
        }

    }
    ######
    
    # save old last one to tail end ->prev
    if (@fc_cmd_exe_history > 0)
    {
        my $last_t = $fc_cmd_exe_history[-1];
        $fc_cmd_exe_history[-1] = $old_last_cml;
        push @fc_cmd_exe_history, $last_t;
    }
    ######

    # add idx for cmd list 
    my $h_idx = 0;
    my @idx_fc_cmd = (); 

    for my $e (@fc_cmd_exe_history)
    {
        my $h_idx_str = sprintf("%3s",$h_idx."");
        my $e_new = "$h_idx_str $e"; 
        push @idx_fc_cmd, $e_new;
        $h_idx++;
    }


    #######

    # output idx+cmd list to cmd txt
    my $fc_string = join "", @idx_fc_cmd; 

    open my $FILE_fn_cmd_exe_history,">" , $fn_cmd_exe_history or die;  
    $FILE_fn_cmd_exe_history->autoflush(1);
    print $FILE_fn_cmd_exe_history $fc_string; 
    $FILE_fn_cmd_exe_history->autoflush(1);
    close $FILE_fn_cmd_exe_history;

    # filter print 
    my @ret_fc_cmd = (); 
    my %h_cmd = (); 

   
    for my $e (reverse @idx_fc_cmd)
    {
        my $old_e = $e; 
        $e =~ s/^.*?\d //;
        $e =~ s/\s+?$//g;

        if ($cml_pat eq "NULL_PAT" or $e =~ m/$cml_pat/i)
        {
            if (not exists $h_cmd{$e} and $e !~ m/^h /) 
            {
                push @ret_fc_cmd, $old_e; 
            }
            $h_cmd{$e}++;
        }
    }

    @ret_fc_cmd = reverse @ret_fc_cmd;

    
    #######
    
    # print out to console
    print join "", @ret_fc_cmd;
    ######

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




