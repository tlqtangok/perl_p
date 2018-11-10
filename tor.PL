#!perl
# written by jidor at 2018-11-10
use strict ;
use feature qw(say);
#use List::Util qw(first sum max shuffle maxstr);
#use List::MoreUtils qw(uniq);
#use File::Spec::Functions;
#use File::Basename qw(dirname basename);
#use Data::Dumper;
#use Scalar::Util qw/reftype/;
#use File::Copy;
#use File::Find;
#use Cwd qw(abs_path);
use strict;
use threads;
use POSIX qw(assert); 
use MIME::Base64 qw( encode_base64 decode_base64);

### global ### 

my $host_name = "algoers.com"; 
my $jd_xx = "NULL"; 
my $jd_incr = "jd_incr"; 
my $b_sz = 1024; # never change !!!
my $b_num = 200; 

my $redis_prefix = qq{redis-cli -h $host_name}; 
my $redis_get_all_keys = qq{$redis_prefix keys '*'}; 
my $redis_incr_jd_incr = qq{$redis_prefix incr $jd_incr}; 
my $redis_del_key_jd_xx = qq{$redis_prefix del $jd_xx}; 
my $redis_del_jd_incr = qq{$redis_prefix del $jd_incr}; 
my $max_len = $b_num * $b_sz;

my $max_jd_incr = 256; 

my $td_num = 8; # for delete threads 
my @td_pool = (); 
my $DEBUG = 0; 
##############
&main();


### sub list ###
sub main()
{#
	&ck_argv(\@ARGV); 

	my @fc = <>;
	my $ret_num = &get_incr_jd_incr(); 
	$jd_xx = "jd_$ret_num"; 
	say $jd_xx; 

	assert(@fc <= $max_len); 

	if ($fc[0] =~ m/[^[:print:][:space:]]/ || $fc[-1] =~ m/[^[:print:][:space:]]/)
	{
		$b_sz = int $b_sz / 10;
		$b_num = $b_num * 10;
	}


	for(my $start = 0; $start <= $max_len; $start += $b_sz)
	{
		my $end = $start + $b_sz -1;
		if ($end >= @fc - 1)
		{
			$end = @fc - 1;
		}; 
		my @e_fc = @fc[$start .. $end]; 
		
		my $e_str_fc = join "", @e_fc; 
		my $cmd_params = &gen_rpush_cmd_params($e_str_fc); 
		&gen_rpush_cmd_and_run($cmd_params); 

		if ($end >= @fc-1)
		{
			last;
		}; 

	}

	#&td_detach(\@td_pool) if @td_pool > 0;
}
### sub list ###
sub ck_argv(\@)
{#
	my $argv_ = shift; 
	die "- no need args\n" if @$argv_ > 0;
}

sub gen_rpush_cmd_params(\@e_str_fc)
{#
	my $e_str_fc = shift; 
	assert($e_str_fc ne ""); 
	return 	encode_base64 $e_str_fc;  
}

sub gen_rpush_cmd_and_run()
{#
	my $cmd_params = shift;
	my $redis_rpush_cmd = qq{$redis_prefix rpush $jd_xx '$cmd_params'}; 
	#say $redis_rpush_cmd; 
	my $max_each_cli =  length $redis_rpush_cmd; 
	die "- each cli param > 127 kb \n" if $max_each_cli > 126 * 1024; 

	my $res_rpush = qx{$redis_rpush_cmd}; 
	die "$res_rpush " if $res_rpush =~ m/ERR/; 
}

sub get_incr_jd_incr()
{#
	my $res = qx{ $redis_incr_jd_incr }; 
	chomp($res); 
	die "redis_incr_jd_incr should return num\n" if ($res !~ m/(\s+)?\d$/); 
	my @arr_t = split m/\s+/,$res; 
	my $cur_num = $arr_t[-1];  
	my $ret_num = $cur_num; 
	if ($cur_num >= $max_jd_incr)
	{
		say "- del all $max_jd_incr jd_xx" if $DEBUG;
		&del_all_jd_xx_restore_jd_incr($cur_num); 
		$ret_num = 0; 
	}

	return $ret_num; 
}

sub del_all_jd_xx_restore_jd_incr($max)
{#
	my $max = shift; 
	qx{$redis_prefix del jd_0}; 

	my @arr_cmd_del = (); 
	for(my $i=1; $i<=$max; $i++)
	{
		my $e_redis_del_key_jd_xx = qq{$redis_prefix del jd_$i}; 
		push @arr_cmd_del,  $e_redis_del_key_jd_xx; 
	}

	push @arr_cmd_del,  qq{$redis_prefix set $jd_incr 0}; 

	my @fc = @arr_cmd_del;
	my $len_fc = @fc; 	
         
	#$td_num = 8;

        my $cml_b_sz = int @fc / $td_num + 1;


	for(my $start = 0; $start < @fc; $start += $cml_b_sz)
	{
		my $end = $start + $cml_b_sz -1;
		if ($end >= @fc - 1)
		{
			$end = @fc - 1;
		}; 
		my @e_fc = @fc[$start .. $end]; 
		
		my $e_cml = join "; ", @e_fc; 

		&td_push(\@td_pool, $e_cml); 

		if ($end >= @fc-1)
		{
			last;
		};

	}

}




sub td_push($td_pool_, $cml)
{#
        my $td_pool_ = shift;

        my $cml = shift;
        my $td =  threads->create
        (
                sub
                {
			#say (	qq{run : $cml}  ); 
			qx($cml); 
                }
        );

        my $len = @$td_pool_;
        $td_pool_->[$len] = $td;
}
sub td_join($td_pool_)
{#
        my $td_pool_ = shift;

        for my $e (@$td_pool_)
        {
                $e->join();
        }

}



=pod
sub td_detach($td_pool_)
{
        my $td_pool_ = shift;

        for my $e (@$td_pool_)
        {
                $e->detach();
        }

}

sub del_redis_key($jd_xx)
{
	my $jd_xx = shift;
	chomp($jd_xx); 
	assert($jd_xx ne ""); 

	my @redis_keys = qx{$redis_get_all_keys};

	for my $e_key (@redis_keys)
	{
		chomp($e_key); 

		if ($e_key =~ m/$jd_xx\"{0,1}$/)
		{
			qx($redis_prefix del $jd_xx);
			last; 
		}
	}
}
=cut
sub test_all_cases()
{
=pod 
change :
	set my $cml_b_sz = 3;
*********************
1.sh
*********************
sh sh.sh 1.txt
sh sh.sh 2.txt
sh sh.sh 3.txt
sh sh.sh 4.txt
sh sh.sh 5.txt

EOF


*********************
sh.sh
*********************
export fn=$1
cat $fn | perl tor.PL
#sleep 1
#exit 0
#cksum  $fn
echo "---"
#sleep 1
perl fr.PL

EOF

=cut

}
