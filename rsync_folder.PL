#!perl

use threads;
### global var ###
#my $cml = 'rsync -avz root@xx.xx.3.14:/root/jd/t/nb/bak_script ./';
#perl -e 'map{ print `date "+%Y%m%d_%H%M"`; $cml="rsync -avz $ENV{bk} /mnt/dev_sdb3/"; system($cml); sleep 10;}(0..10e6);'


my @cml_arr = (
	"rsync -avz /home/ww/data/jd/bak_script /mnt/dev_sdb3/jd/bak_script", 
	"rsync -avz /home/ww/data/jd/t/dl /mnt/dev_sdb3/jd/dl"
);


###

&main(); 

### sub list ###
sub main()
{
	my $td_0 = threads->create(

			sub {
				while (true)
				{
					print `date "+%Y%m%d_%H%M"`;

					for my $e_cml (@cml_arr)
					{
						system($e_cml);

					}

					sleep 1*60*60;
				}

			}


			);

	print "- bg run :[ $cml ]";
	$td_0->join();
}



