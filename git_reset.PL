#!perl


my $master_or_main = qx(git branch); 
$master_or_main =~ s/^.* (.*)$/\1/;
chomp($master_or_main);
my $cmd_reset_pull = qq(
set -e
git fetch --all
git reset --hard origin/$master_or_main
git pull
); 


#system( $cmd_reset_pull ); 
printf( $cmd_reset_pull );


