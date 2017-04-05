#!perl
use feature qw(say); 

@auto_txt=`cat /home/linqi/perl_p/.auto_makefile.txt `;
$perc_no=21; 
$per_c=@auto_txt[$perc_no++];# %.c  
$CC= @auto_txt[$perc_no++];# \t$(CC)
$per_cpp= @auto_txt[$perc_no++]; # %.cpp

@c_files = `ls *.c 2>/dev/null`; 
@cpp_files = `ls *.cpp 2>/dev/null`; 
$len_cpp=@cpp_files; 
$len_c=@c_files;


if($len_c){
@arr_deps=(); 
while($len_c--){
@arr=`ccac -MM @c_files[$len_c]`;

@arr=grep ! m/(\(LLVM)|(\(c\))/  , @arr; 
map{chomp;s/\\/ /g;s/\s{2,}/ /g;}@arr; 
my $deps="@arr";
$deps =~ s/\s{1,}/ /g;
push @arr_deps,$deps."\n"; 


}# --- END $len_c 


}


 if($len_cpp){
@arr_deps=(); 
while($len_cpp--){
@arr=`ccac -MM @cpp_files[$len_cpp]`; 

@arr=grep ! m/(\(LLVM)|(\(c\))/  , @arr; 
map{chomp;s/\\/ /g;}@arr; 
my $deps="@arr";
$deps =~ s/\s{1,}/ /g;
push @arr_deps,$deps."\n"; 

}# --- END $len_cpp 


}
if ( !$len_c && !$len_cpp){
	say "no c and cpp files !"; 
}
#say @arr_deps; 

@arr_out=(); 
$i=0; 
for($i=0;$i <21;$i++){
push @arr_out , @auto_txt[$i]; 
}

map{
push @arr_out, $_; 
push @arr_out, $CC;  
}@arr_deps; 
for($i=25; $i<@auto_txt; $i++){
push @arr_out,@auto_txt[$i]; 
}
say @arr_out; 



