#!perl 
use strict ;
use feature qw(say);
use List::Util qw(first sum max shuffle maxstr);
#use List::MoreUtils qw(uniq);
use File::Spec::Functions;
use File::Basename qw(dirname basename);
use Data::Dumper;
use Scalar::Util qw/reftype/;
use File::Copy;


my $src_folder = q(/home/ubuntu/jd/t/t0);

my $full_fn_of_list_file = $src_folder . q(/1.win.txt);
my $dst_folder = q(/home/ubuntu/jd/t/t1);


open my $FP_fn_list_all, "<", $full_fn_of_list_file; 
my @fn_all = <$FP_fn_list_all>; 
close $FP_fn_list_all;






=pod
@fn_all =
qw(
E:\jd\t\bgi_gpu_server\1
E:\jd\t\bgi_gpu_server\Makefile
E:\jd\t\bgi_gpu_server\README.md
E:\jd\t\bgi_gpu_server\test.PL
E:\jd\t\bgi_gpu_server\include\bwt.h
E:\jd\t\bgi_gpu_server\include\zlog.h
);
=cut
chomp(@fn_all);


my @prev = (); 
my $prev_mtime = "";
map{
	my @files = (sort{ -M "$a" <=> -M "$b" } @fn_all)[0..3];

	if (@prev > 0)
	{
		

			my $files_mtime = &gen_size($files[0]);
			#say $files_mtime;

			if ($files_mtime != $prev_mtime)
			{

				my ($original, $new_copy) = &gen_copy_cmd($files[0], $dst_folder);
				my $ts = &get_timestamp();
				#say $ts; 
				say("- copy\t", $original," ", $new_copy);
				copy( $original, $new_copy) or die "Copy failed: $!";        # new_copy can be a dir 
			}

	}
	@prev = (@files); 
	$prev_mtime = &gen_size($prev[0]);

	sleep(4);
}(0..1e6);



### sub list ###
sub gen_copy_cmd($fn, $dst_folder)
{
	my $fn = shift; 
	my $fn_r0 = $fn; 
	$fn =~ s/.*\/t0/$dst_folder/;
	return $fn_r0, $fn; 
}

sub gen_size($)
{
	my $fn = shift; 
	my $mtime = (stat($fn))[9];
	return $mtime;
}

### perl timestamp ### 
sub add_zero_if_lt_10($){
	my $num = shift @_; 
	if ($num >=10 ){
		return "$num" 
	}
	return "0$num"; 
}
sub get_timestamp(){

# perl get time, for second:
 #perl -e 'print $^T '
#1477400615
my $if_windows = "True"; 

    my $time_stamp = `date "+%Y%m%d_%H%M"` if $if_windows ne "True"; 

    if ( $if_windows eq "True"){
        my ($second, $minute, $hour, $date, $month, $yearOffset, $dayOfWeek, $dayOfYear, $daylightSavings) = localtime();
        # 17 16 20 24 7 116 3 236 1
        # 2016-08-24 20:16

        my $year = 1900 + $yearOffset;
        $month = $month + 1 ; 
        $month = &add_zero_if_lt_10($month); 
        $date = &add_zero_if_lt_10($date); 
        $hour = &add_zero_if_lt_10($hour); 
        $minute = &add_zero_if_lt_10($minute); 
        $time_stamp = "$year$month${date}_${hour}$minute"; 
    }

    chomp( $time_stamp ); 
    return $time_stamp; 

}


#################

 # /home/ubuntu/jd/t/t0 ./
 # /home/ubuntu/jd/t/t1 ./
