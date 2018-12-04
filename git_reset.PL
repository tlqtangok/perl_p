#!perl

my $cmd_reset_pull = qq(
set -e
git fetch --all
git reset --hard origin/master
git pull
); 


system( $cmd_reset_pull ); 


