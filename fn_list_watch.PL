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


my $src_folder = q(E:\jd\t\bgi_gpu_server);
my $dst_folder = q(Y:\git\bgi_gpu_server);
 
my $cmd_read_win = q(type E:\jd\t\bgi_gpu_server\1.win.txt);   # this is file list 
my @fn_all = `$cmd_read_win`;

=pod
@fn_all =
qw(
E:\jd\t\bgi_gpu_server\1
E:\jd\t\bgi_gpu_server\Makefile
E:\jd\t\bgi_gpu_server\README.md
E:\jd\t\bgi_gpu_server\test.PL
E:\jd\t\bgi_gpu_server\include\bwt.h
E:\jd\t\bgi_gpu_server\include\zlog.h
);
=cut
chomp(@fn_all);


my @prev = (); 
my $prev_mtime = "";
map{
	my @files = (sort{ -M "$a" <=> -M "$b" } @fn_all)[0..3];

	if (@prev > 0)
	{
		

			my $files_mtime = &gen_size($files[0]);

			if ($files_mtime != $prev_mtime)
			{

				my ($original, $new_copy) = &gen_copy_cmd($files[0], $dst_folder);
				say($original," ", $new_copy);
				copy( $original, $new_copy) or die "Copy failed: $!";        # new_copy can be a dir 
			}

	}
	@prev = (@files); 
	$prev_mtime = &gen_size($prev[0]);

	sleep(5);
}(0..1e6);



### sub list ###
sub gen_copy_cmd($fn, $dst_folder)
{
	my $fn = shift; 
	my $fn_r0 = $fn; 
	$fn =~ s/.*bgi_gpu_server/$dst_folder/;
	return $fn_r0, $fn; 
}

sub gen_size($)
{
	my $fn = shift; 
	my $mtime = (stat($fn))[9];
	return $mtime;
}

#################


