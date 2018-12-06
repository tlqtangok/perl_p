#!perl


use Data::Dumper; 



my @argv = ( @ARGV );
my $templ_ = q{ 
	cat __FILELIST__ | perl -e '@arr=<>;map{__MAPSTATEMENT__;print ; }@arr;'
}; 

if (@argv <= 1 ){
	print "- please run by:\n\t";
	print "map_ txt.txt 's/abc/ABC/;'\n" ; 
	print $templ_; 
	exit (1); 
}


my $__MAPSTATEMENT__ = @argv[-1]; 

@argv[-1] = ""; 

my $__FILELIST__ = "@argv"; 


$templ_ =~ s/__FILELIST__/$__FILELIST__/; 
$templ_ =~ s/__MAPSTATEMENT__/$__MAPSTATEMENT__/; 

=pod
print '- run:',"\n";
print '===================', ""; 
print $templ_ ; 
print '===================', "\n\n"; 
=cut

system(  $templ_ ); 

#print "EOF\n";

