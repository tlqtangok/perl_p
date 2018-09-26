#perl

$| = 1;
map
{
        print "$_";
        sleep(222);
}(0..1e6);

