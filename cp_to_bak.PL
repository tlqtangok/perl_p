#perl
use File::Copy;
use feature qw(say); 
use File::Basename qw(dirname basename);
use File::Spec;

my $SP = File::Spec->catfile('', ''); 


my $if_windows = 'False';
$if_windows = "True"  if @ENV{WINDIR};

my $time_stamp =&get_timestamp();
my @ARGV_ = ( @ARGV );

for $file ( @ARGV_)
{

    if (-d $file && $if_windows eq "True" ){
        
        $_ = $file;
        chomp; 
        my $_old_fn = $_; 
        my $dir_name = dirname($_); 
        my $base_name = basename($_); 

        my $new_copy = 'bak.'.$base_name.".$time_stamp";
        $new_copy = $dir_name . $SP . $new_copy if $dir_name ne ".";

        system( " rmdir /s /q $new_copy > %tmp%\log.log 2>&1");
        my $bat_cmd = qq( mkdir $new_copy && xcopy /V /Y /Q /E $_ $new_copy > %tmp%\log.log 2>&1 );
        system( $bat_cmd );
        print $new_copy,"\n";
    }

    elsif ( ($file =~ m/\w/) &&  ( $if_windows eq "True" ) ) {
        my @arr = glob $file;
#print @arr ;
        my @ARGV=(@arr);
        if (-e $file && $if_windows eq "True" ){

            map{
            chomp;
            my $_old_fn = $_; 

            my $dir_name = dirname($_); 
            my $base_name = basename($_); 
            
            my $new_copy = 'bak.'  .$base_name.  ".$time_stamp";
            $new_copy = $dir_name ."$SP". $new_copy if $dir_name ne "."; 
            unlink $new_copy if -e $new_copy;
            copy( $_old_fn, $new_copy ) or die "Copy failed: $!";
            print " ", $new_copy, "\n";
            }($file);

        }
        else {
            die "- no such file match $file\n";
        }
    }

    else {
        if ($file eq "" ){
            die "- $file is filename or foldername.\n";
        }
        else {
            die "- Linux is not supported\n";
        }

    }




}



### sub list ###
sub add_zero_if_lt_10($num){
        my ($num) = (@_[0]);
        chomp($num);
        if ( $num < 10 ){
                return "0$num";
        }
        else {
                return "$num";
        }
}

sub get_timestamp(){
        my $if_windows = 'False';
        $if_windows = "True"  if @ENV{WINDIR};



        my $time_stamp = `date "+%Y%m%d%H%M"` if $if_windows ne "True";

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
                $time_stamp = "$year$month${date}${hour}$minute";
        }

        chomp( $time_stamp );
        return "".$time_stamp;

}
