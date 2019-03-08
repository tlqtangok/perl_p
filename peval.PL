#!perl 
use feature qw(say);
while(my $e_l = <>)
{
   chomp(); 
   last   if (m/^q$|^quit$/i);
   eval "\$_=$e_l;"; 
   say $_;
}
