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
	my $num_topn = 11; 
	if (@ARGV > 0)
	{
		$num_topn = $ARGV[0]; 	
	}

	#cat fn_all.txt |perl -e '@a=<>;chomp(@a);@aa=sort {-s $b <=> -s $a} @a;print "@aa[0..10]";'
	find(\&save_g_fn_list ,  "$curdir"); 

    #my %hash_fn_sz = (); 
    #my $cmd_eval_to_hash =  q{%hash_fn_sz = } . "qw{$g_fn_list}"; 
    #eval($cmd_eval_to_hash);



	my @fn_modified =  sort { $hash_fn_sz{$b} <=> $hash_fn_sz{$a} } keys %hash_fn_sz;

    #print join "\n", @fn_modified[0..$num_topn];
    my $cnt = 0;
    for( @fn_modified )
    {
       print $_, " || ", $hash_fn_sz{$_}, "\n"; 
       if ($cnt > $num_topn) 
       {
           last;
       }
       $cnt++;
    }
    print "\n"; 
}

sub save_g_fn_list()
{
        my $fn = $File::Find::name;
        if( &if_fn_match_pattern($fn, \@filter_param_list))
        {
			# $g_fn_list .= $fn. " " . (stat($fn))[9]."\n"; # modify time

                $g_fn_list .= $fn. " " . (stat($fn))[7]."\n"; # size
                $hash_fn_sz{$fn} = (stat($fn))[7];
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
				if(-f $fn && $fn !~ m/\.git/)
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
				if(-f $fn && $fn !~ m/\.git/)
				{
					$ret_code = 1; 
				}

				last;
			}
		}
	}

	return $ret_code;
}
