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
use File::Find;
use Cwd qw(abs_path);
use strict;
use threads;

&main();




### sub list ###

sub main()
{

	my $port_keywords = "102"; 
	my $ff = qq{perl $ENV{perl_p}/ff.PL};
	my @all_port_list = `netstat -anop|grep $port_keywords| $ff 3`;
	my %port_pure = (); 

	for my $e_port (@all_port_list)
	{
		chomp($e_port);
		$e_port =~ s/^.*\://g;
		$port_pure{$e_port}++; 
	}

	my @ports = sort keys %port_pure;

	my $flag_split_once = 0;
	for my $e_port (@ports)
	{

		#say join "\n", @ports; 
		if( $e_port >= 10250 && $flag_split_once++ == 0)
		{
			say ""; 
		}

		say $e_port;
		

	}

}

