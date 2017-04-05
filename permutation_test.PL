###-Usage-:  perl _perl.PL 2>/dev/null ###
#!perl
use feature qw(say);

### CONFIG AREA ###
if ( @ARGV == 0 ){
die "- please input num\n";
}
my $MAX = @ARGV[0];
####### END #######

my @a=qw();
my $len_a=$MAX*2-1;
my $file_header='
#!perl
use feature qw(say);
#--- TEST ---
my $loc_i=0;
my $MAX='."$MAX".'; # the num you want to make the
my @arr_loc=qw();
#---start permutation
';
my $for_header='for(@arr_loc[$loc_i]=0;@arr_loc[$loc_i]<$MAX; @arr_loc[$loc_i]++){      next if $arr_loc[$loc_i] ~~ @arr_loc[0..$loc_i-1];       $loc_i++;       #---;
'x ($MAX-1);
#my $say_kernel='     for(@arr_loc[$loc_i]=0;@arr_loc[$loc_i]<$MAX; @arr_loc[$loc_i]++){ next if $arr_loc[$loc_i] ~~ @arr_loc[0..$loc_i-1]; say @arr_loc ;  }#end
my $say_kernel='     for(@arr_loc[$loc_i]=0;@arr_loc[$loc_i]<$MAX; @arr_loc[$loc_i]++){ next if $arr_loc[$loc_i] ~~ @arr_loc[0..$loc_i-1];  ;  }#end
' ;
my $for_tail='$loc_i--;      }    #end for
' x ($MAX-1) ;


open FP, '>', "bak.perl.PL";
select FP;
say "$file_header","$for_header\n","$say_kernel\n","$for_tail";
close FP;
select STDOUT;

my @perl_cml=qw(perl bak.perl.PL);
open STDERR, ">",'/dev/null';


system(@perl_cml);

close STDERR;
`rm bak.perl.PL`;

