CC = g++ -std=c++11 -Wno-format
c_flag = -g -o
c_files =  \
$(wildcard *.cpp) $(wildcard *.c)
h_files = \
$(wildcard *.hpp) $(wildcard *.h)
o_files_mess = \
$(patsubst %.cpp,%.o,${c_files}) $(patsubst %.c,%.o,${c_files})
o_files = \
$(filter %.o, $(o_files_mess))
elf_file=  \
mainapp.exe

rm_files = \
*.o *.dep *.elf  *.s *.exe


all:$(elf_file)

$(elf_file):$(o_files)
	$(CC) $(c_flag) $@ $(o_files)
main.o: main.cpp
	$(CC) $(c_flag) $@ -c $<
#	@echo "c_file is:$(c_files)"; echo "h_files:$(h_files)"; echo "o_files:$(o_files)"

clean:
	rm -rf $(rm_files)

run:$(elf_file)
	./$(elf_file)

