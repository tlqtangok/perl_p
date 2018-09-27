#!perl
# written by jd 
use strict; 
use threads; 
use POSIX qw(assert); 
use feature qw(say);

if ("__main__" eq "__main__")
{

	my %hash_listen_port_start_cml = 
	(
		4443,		"./ngrokd -tlsKey=server.key -tlsCrt=server.crt -domain=lewelab.com -httpAddr=:10242 -httpsAddr=:10243", 
		4442,		"./ngrokd -tlsKey=server.key -tlsCrt=server.crt -domain=lewelab.com -tunnelAddr=:4442 -httpAddr=:10244 -httpsAddr=:10245",
		4441,		"./ngrokd -tlsKey=server.key -tlsCrt=server.crt -domain=lewelab.com -tunnelAddr=:4441 -httpAddr=:10248",
		10249,		"/root/anaconda3/bin/jupyter notebook --allow-root",
		10241,		"perl /root/t/git/ngrok/_tell_port.PL",
	); 

	my %hash_port_status_check_cml = 
	(
		4443,		" netstat -anop |grep 4443 | grep ngrokd| grep LISTEN ", 
		4442,		" netstat -anop |grep 4442 | grep ngrokd| grep LISTEN ",
		4441,		" netstat -anop |grep 4441 | grep ngrokd| grep LISTEN ",
		10249, 		" netstat -anop |grep 10249| grep python| grep LISTEN ", 
		10241,		" pgrep -f _tell_port -al && netstat -anop |grep 10241 | grep docker",
	);


	while(1)
	{
		my @td_pool = (); 
		my $perl_p = $ENV{perl_p};

		for my $port (keys %hash_listen_port_start_cml)
		{
			&td_push(\@td_pool, \%hash_listen_port_start_cml, \%hash_port_status_check_cml, $port);
		}
		&td_detach(\@td_pool);

		sleep(20 * 60 * 60);
	}

}


### sub list ###

sub td_join($td_pool_){
	my $td_pool_ = shift;

	for my $e (@$td_pool_)
	{
		$e->join();
	}

}


sub td_detach($td_pool_){
	my $td_pool_ = shift;

	for my $e (@$td_pool_)
	{
		$e->detach();
	}

}


sub td_push($td_pool_, $hash_listen_port_start_cml_, \%hash_listen_port_start_cml_, $port)
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
				say "- no procedure, now start $cml on $port ..."; 
				# system("$cml");
			}
		}
	);

	my $len = @$td_pool_;
	$td_pool_->[$len] = $td;
}

sub chk_ngrok_running_status(\%hash_listen_port_start_cml_, \%hash_port_status_check_cml_, $port )
{
	my $hash_listen_port_start_cml_ = shift; 
	my $hash_port_status_check_cml_ = shift;
	my $port = shift;

	my $hash_chk_cmd = $hash_port_status_check_cml_->{$port}; 

	qx{ $hash_chk_cmd };  # must on LISTEN status

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

