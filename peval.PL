#!perl 
use feature qw(say);
while(<>)
{
   chomp(); 
   last   if (m/^q$|^quit$/i);
   say eval $_; 
}
