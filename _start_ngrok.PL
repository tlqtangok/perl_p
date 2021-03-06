#!perl
# written by jd
use strict;
use threads;
use POSIX qw(assert);
use feature qw(say);

if ("__main__" eq "__main__")
{
		&chk_only_one_perl_start();

        my %hash_listen_port_start_cml =
        (
			 "4443 + 10240, ssh",           " ./ngrok  -config=ngrok.yml start ssh & ",
			"4443 + 10242, http",           " ./ngrok  -config=ngrok.yml start http & ",
			     "0 + 10242, nb",           " jupyter-notebook & ",
		   "samba, mount aliyun",			" sudo mount -t cifs -o username=root,password=jd,uid=1000,gid=1000 //119.23.8.57/t /home/pi/jd/t/nb/lnk & ",
	  "rsync yun nb to local pi",			" rsync -avz /home/pi/jd/t/nb/lnk/nb /home/pi/jd/t/nb/yun ",
        );

        my %hash_port_status_check_cml =
        (
			"4443 + 10240, ssh",			" pgrep ngrok -al | grep ssh ",
		   "4443 + 10242, http",			" pgrep ngrok -al | grep http ",
	   			"0 + 10242, nb",			" pgrep -f jupyter -al ",
		  "samba, mount aliyun",			" sudo df -k |grep 119 ",
	 "rsync yun nb to local pi",			" pgrep rsync ",
        );

		my $cnt_loop = 0; 

        while(1)
		{
			my $time_stamp = `date "+%Y%m%d_%H%M"`;
			
			print "- loop $cnt_loop at $time_stamp"; 
			my @td_pool = ();

			for my $port (keys %hash_listen_port_start_cml)
			{
				&td_push(\@td_pool, \%hash_listen_port_start_cml, \%hash_port_status_check_cml, $port);
			}

			&td_join(\@td_pool);

			sleep(20 * 60);
			$cnt_loop++; 
		}

}




### sub list ###
sub chk_only_one_perl_start()
{
        my $this_script = __FILE__;
        my @cmd_ans_this_perl = `pgrep -f $this_script -al`;
        if( @cmd_ans_this_perl >= 2)
        {
           print @cmd_ans_this_perl;
           die "- more than one perl $this_script is running\n";
        }
}
sub td_join($td_pool_)
{
        my $td_pool_ = shift;

        for my $e (@$td_pool_)
        {
                $e->join();
        }

		sleep 4;
}


sub td_detach($td_pool_)
{
        my $td_pool_ = shift;

        for my $e (@$td_pool_)
        {
                $e->detach();
        }

}


sub td_push()
{
        my $td_pool_ = shift;
        my $hash_listen_port_start_cml_ = shift;
        my $hash_port_status_check_cml_ = shift;
        my $port = shift;

        my $cml = $hash_listen_port_start_cml_->{$port};

        my $td =  threads->create
        (
                sub
                {
                        if (&chk_ngrok_running_status($hash_listen_port_start_cml_, $hash_port_status_check_cml_, $port) != 0)
						{
							sleep  int(rand(5 * 4));
							say "- now start $cml on $port ...";
							system("$cml");
						}
                }
        );

        my $len = @$td_pool_;
        $td_pool_->[$len] = $td;
}

sub chk_ngrok_running_status()
{
        my $hash_listen_port_start_cml_ = shift;
        my $hash_port_status_check_cml_ = shift;
        my $port = shift;

        my $hash_chk_cmd = $hash_port_status_check_cml_->{$port};

        qx{ $hash_chk_cmd };  

        my $rc = $? ;

        if ($rc != 0)
        {
                say "- no $port : $hash_listen_port_start_cml_->{$port} is running ";
        }
        else
        {
                say "- running on $port : $hash_listen_port_start_cml_->{$port} ";
        }

        return $rc;
}


