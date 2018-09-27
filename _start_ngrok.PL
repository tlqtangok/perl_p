#!perl
# written by jd 2018-09-11
use strict; 

use threads;
use feature qw(say);




if ("main" eq "main")
{
	my $this_script = __FILE__; 
	my @cmd_ans_this_perl = `pgrep -f $this_script -al`;
	if( @cmd_ans_this_perl >= 2)
	{
	   print @cmd_ans_this_perl; 
	   die "- more than one perl $this_script is running\n";
	}	

	my $cmd_ngrok_ssh  = q( ./ngrok   -config=ngrok.yml start ssh  );
	my $cmd_ngrok_http = q( ./ngrok  -config=ngrok.yml start http );
	my $cmd_jupyter =    q( jupyter-notebook  );
	my $cmd_mount_lewelab = q( sudo mount -t cifs -o username=root,password=jd,uid=pi,gid=pi,umask=0022 //119.23.8.57/t /home/pi/jd/t/nb/lnk );
	

	my $cmd_stat_ssh  =       q( pgrep ngrok -al | grep ssh  ) ;
	my $cmd_stat_http =       q( pgrep ngrok -al | grep http ) ;
	my $cmd_stat_jupyter =    q( pgrep jupyter -al );
	my @arr_td = (0,1,2,3);
	my $cnt_td = 0; 


	my @cmd_ans_aliyun_mounted = qx( sudo df -k |grep 119 ); 
	if (@cmd_ans_aliyun_mounted == 0)
	{
		say("run :", $cmd_mount_lewelab);	
		system($cmd_mount_lewelab);	
	}



	while(1)
	{
		system( q(date "+%Y%m%d%H%M") );
		
		my @cmd_ans_of_stat_ssh  = ` $cmd_stat_ssh  `;
		if (@cmd_ans_of_stat_ssh >= 1)
		{
			print "- already has ngrok ssh client\n";
		}
		else
		{
			@arr_td[$cnt_td++] = threads->create(
				sub 
				{
					say("run :",$cmd_ngrok_ssh); 
					system($cmd_ngrok_ssh); 
				}
			);
		}


		my @cmd_ans_of_stat_http = ` $cmd_stat_http `;
		if (@cmd_ans_of_stat_http >= 1)
		{
			print "- already has ngrok http client\n";
		}
		else
		{
			@arr_td[$cnt_td++] = threads->create(
				sub 
				{ 
					say("run :", $cmd_ngrok_http);
					system($cmd_ngrok_http);
				}
			);
		}

		my @cmd_ans_of_stat_jupyter  = ` $cmd_stat_jupyter  `;
		if( @cmd_ans_of_stat_jupyter >= 1)
		{
			print "- already has jupyter process\n";
		}
		else
		{
			@arr_td[$cnt_td++] = threads->create(
				sub 
				{ 
					say("run :", $cmd_jupyter);
					system($cmd_jupyter);
				}
			);
		}

		@arr_td[$cnt_td++] = threads->create(
			sub 
			{ 
				my $cmd_rsync = q( rsync -avz /home/pi/jd/t/nb/lnk/nb /home/pi/jd/t/nb/yun );
				say("run :", $cmd_rsync); 
				system($cmd_rsync); 
			}
		);

		if ($cnt_td)
		{
			for my $td (@arr_td[0..$cnt_td-1])
			{
				$td->detach();
			}
		}

		$cnt_td = 0;

		sleep(2 * 60 * 60);  # sleep 2 hour
	}

}


### sub list ###



