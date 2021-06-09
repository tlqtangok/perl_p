#!perl

use threads;
### global var ###
my $cml = 'rsync -avz root@xx.xx.3.14:/root/jd/t/nb/bak_script ./';
###

&main(); 

### sub list ###
sub main()
{
	my $td_0 = threads->create(

			sub {
				while (true)
				{
					system($cml);
					sleep 1*60*60;
				}

			}


			);

	print "- bg run :[ $cml ]";
	$td_0->join();
}



