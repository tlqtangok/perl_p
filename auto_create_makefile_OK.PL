#!perl
use strict;
use feature qw(say); 



my $CC_NAME = "g++";
my $perl_p = @ENV{perl_p};
######## fc of auto_txt_str ###########
##### NEVER EDIT FOLLOWING STRING #####
#######################################

my $auto_txt_str =q{CC = g++ -std=c++11 

nullstring := 
dep_I_incpath = -I./include/hiredis -I./include
dep_L_libpath = -L./lib
dep_l_libname = -lhiredis
#deps_extra = $(dep_I_incpath) $(dep_L_libpath)  $(dep_l_libname)

deps_extra = $(nullstring)

#c_flag = -g -Wno-format -fpermissive 
c_flag = -g -pthread -Wno-format -fpermissive

c_files =  \
$(wildcard *.cpp) $(wildcard *.c) $(wildcard *.cc)
h_files = \
$(wildcard *.hpp) $(wildcard *.h)
o_files_mess = \
$(patsubst %.cpp,%.o,${c_files}) $(patsubst %.c,%.o,${c_files}) $(patsubst %.cc,%.o,${c_files})
o_files = \
$(filter %.o, $(o_files_mess))
elf_file=  \
mainapp.exe

rm_files = \
*.o *.dep *.elf  *.s *.exe


all:$(elf_file)

$(elf_file):$(o_files)
	$(CC) $(c_flag) -o $@ $(o_files) $(deps_extra) 
%.o: %.c $(h_files)
	$(CC) $(c_flag) -o $@ -c $< $(deps_extra) 
%.o: %.cpp $(h_files)
	$(CC) $(c_flag) -o $@ -c $< $(deps_extra) 
%.o: %.cc $(h_files)
	$(CC) $(c_flag) -o $@ -c $< $(deps_extra) 
#	@echo "c_file is:$(c_files)"; echo "h_files:$(h_files)"; echo "o_files:$(o_files)"

clean:
	rm -rf $(rm_files)

run:$(elf_file)
	./$(elf_file)

};

##### end fc of auto_txt_str ########

my @auto_txt=split m/\n/, $auto_txt_str;
map{$_.="\n";}@auto_txt;
my $cnt_met_CC = 0;
for (0..@auto_txt -1){$cnt_met_CC++; last if $auto_txt[$_] =~ m/^\s+\$\(CC\)/; };

my $perc_no = "NULL"; 

$perc_no= $cnt_met_CC ;
my $perc_no_R0 = $cnt_met_CC;



my $per_c=@auto_txt[$perc_no++];# %.c  



my $CC= @auto_txt[$perc_no++];# \t$(CC)
my $per_cpp= @auto_txt[$perc_no++]; # %.cpp


my @c_files = `ls *.c 2>/dev/null`; 
chomp(@c_files); 

my @cpp_files = `ls *.cpp 2>/dev/null`; 
chomp(@cpp_files);

my @cc_files = `ls *.cc 2>/dev/null`; 
chomp(@cc_files);

my $len_cpp=@cpp_files; 
my $len_c=@c_files;
my $len_cc=@cc_files;

my @arr_deps = (); 

if($len_c)
{
	#@arr_deps=(); 
	while($len_c--)
	{
		my @arr=`$CC_NAME -std=c++11 -MM @c_files[$len_c]`;
		@arr=grep ! m/(\(LLVM)|(\(c\))/  , @arr; 
		map{chomp;s/\\/ /g;s/\s{2,}/ /g;}@arr; 

		my $deps="@arr";
		$deps =~ s/\s{1,}/ /g;
		push @arr_deps,$deps."\n"; 
	} 
}



if($len_cpp)
{
	#@arr_deps=(); 
	while($len_cpp--){

		my @arr=`$CC_NAME -std=c++11 -MM @cpp_files[$len_cpp]`; 

		@arr=grep ! m/(\(LLVM)|(\(c\))/  , @arr; 
		map{chomp;s/\\/ /g;}@arr; 
		my $deps="@arr";
		$deps =~ s/\s{1,}/ /g;

		push @arr_deps,$deps."\n"; 
	} # --- END $len_cpp 
}

if($len_cc)
{
	#@arr_deps=(); 
	while($len_cc--)
	{
		my @arr=`$CC_NAME -std=c++11 -MM @cc_files[$len_cc]`;

		@arr=grep ! m/(\(LLVM)|(\(c\))/  , @arr; 
		map{chomp;s/\\/ /g;s/\s{2,}/ /g;}@arr; 

		my $deps="@arr";
		$deps =~ s/\s{1,}/ /g;
		push @arr_deps,$deps."\n"; 
	} 
}




if ( !$len_c && !$len_cpp && !$len_cc)
{
	say "no c, cpp, cc files !"; 
}
#say @arr_deps; 

my @arr_out=(); 
my $i=0; 

for($i=0;$i <$perc_no_R0;$i++)
{
	push @arr_out , @auto_txt[$i]; 
}

map
{
	push @arr_out, $_; 
	push @arr_out, $CC;  
}@arr_deps; 
for($i=$perc_no_R0 + 4 + 2; $i<@auto_txt; $i++)
{
	push @arr_out,@auto_txt[$i]; 
}
say @arr_out; 

