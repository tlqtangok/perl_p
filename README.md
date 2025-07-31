# perl_p

list all:

```

t, 
lsh,
lshp,
ff,
fsrc,
peval,
s_b,
findf_by_name，
perl_create_makefile,
cp_to_bak,
see_path,
tol,
h,
!
torel,
full,
fullw,
see_path,
grepw, 
lst,
cbin,
p,
pd,
gl,
tarc,
tart,
tarx,
peval,
lshp,
ls_big_fn,
repl,
lsd,
ecd,
ecds,
```

`ai generated doc, may have error! ` 
===

# perl_p - Utilities for Enhancing Effectiveness

A collection of command-line utilities built with Perl, Python, C++, and batch scripts to enhance productivity and effectiveness in daily development tasks.

## Project Overview

**Language Composition:**
- Perl (34.2%) - Core scripting utilities
- Python (24.4%) - Advanced processing tools
- C++ (24.3%) - Performance-critical components
- MATLAB (5.3%) - Mathematical computations
- Batchfile (4.5%) - Windows automation
- Shell (3.0%) - Unix/Linux support
- Other (4.3%) - Miscellaneous utilities

## Command-Line Tools

### 1. lsh - List with Human-readable Sizes
Enhanced file listing with formatted sizes and timestamps.
```bash
lsh
# Shows files with formatted time and human-readable sizes
# drwxr-xr-x  4 user group  128 09:30 documents/
# -rw-r--r--  1 user group 1.2K 08:45 readme.txt

lsh 10
# Shows first 10 files only

lsh /path/to/dir
# Lists specific directory
```

### 2. lst - List Today's Files
Lists executable files (.exe, .dll) modified today.
```bash
lst
# Output shows files modified on current date:
# 2025-07-31
# -rw-r--r-- program.exe
# -rw-r--r-- library.dll

lst /other/path
# Lists today's executables in specified path
```

### 3. grepw - Enhanced Grep with Word Boundaries
Grep with color support and word boundary matching.
```bash
grepw "function" *.pl
# Finds whole word "function" in Perl files with color highlighting

echo "test function call" | grepw "function"
# Output: test function call (with "function" highlighted)
```

### 4. cbin - Convert to Binary
Converts numbers to binary format through Perl processing.
```bash
cbin 255
# Processes through cbin.PL and outputs binary representation

cbin 0xFF
# Handles hexadecimal input conversion
```

### 5. p - Print/Process Utility
Quick text processing and printing utility.
```bash
p "Hello World"
# Quick print functionality

p < input.txt
# Process text from file
```

### 6. pd - Push Directory
Directory navigation with stack management.
```bash
pd /home/user/projects
# Changes to directory and maintains navigation stack

pd
# Shows current directory stack
```

### 7. tarc - Create Tar Archive
Creates compressed tar archives.
```bash
tarc backup.tar.gz *.txt *.log
# Creates compressed archive with specified files

tarc project.tar.gz src/ docs/
# Archives entire directories
```

### 8. tarx - Extract Tar Archive
Extracts tar archives with automatic format detection.
```bash
tarx backup.tar.gz
# Extracts all contents from archive

tarx project.tar.gz /destination/
# Extracts to specific location
```

### 9. tart - List Tar Contents
Lists contents of tar archives without extraction.
```bash
tart backup.tar.gz
# Shows archive contents:
# src/main.c
# docs/readme.txt
# config/settings.ini
```

### 10. h - Enhanced History
Smart command history with filtering and formatting.
```bash
h
# Shows formatted command history

h grep
# Shows history entries containing "grep"
# Uses h_linux.PL for enhanced formatting
```

### 11. tol - To Lowercase
Converts text or filenames to lowercase.
```bash
tol "HELLO WORLD"
# Output: hello world

echo "MixedCase.TXT" | tol
# Output: mixedcase.txt
```

### 12. ff - Find Files
Enhanced file finding with pattern matching.
```bash
ff "*.pl" /home/user
# Finds all Perl files recursively

ff "main.c" .
# Finds main.c in current directory tree
```

### 13. env_set - Environment Variable Setter
Sets and manages environment variables.
```bash
env_set PATH "/usr/local/bin:$PATH"
# Updates PATH environment variable

env_set PERL_P "/home/user/perl_p"
# Sets custom environment variable
```

### 14. peval - Perl Expression Evaluator
Evaluates Perl expressions from command line.
```bash
peval "2 + 3 * 4"
# Output: 14

peval "length('hello world')"
# Output: 11

peval "join(',', 1..5)"
# Output: 1,2,3,4,5
```

### 15. gl - Git Log Enhanced
Enhanced git log with custom formatting.
```bash
gl
# Shows enhanced git log

gl --oneline -10
# Shows last 10 commits in compact format
```

### 16. full - Full Path Display
Shows absolute paths of files.
```bash
full *.txt
# Output: /home/user/project/file1.txt
#         /home/user/project/file2.txt

full .
# Shows full path of current directory
```

### 17. fullw - Full Path Windows Format
Shows full paths in Windows format.
```bash
fullw *.txt
# Output: C:\Users\user\project\file1.txt
#         C:\Users\user\project\file2.txt
```

### 18. gs - Git Status Enhanced
Smart git status with add command suggestions.
```bash
gs
# Shows modified files and suggests git add command:
# M  file1.txt
# M  file2.cpp
#
# git add file1.txt file2.cpp
```

### 19. lshp - List with Permissions
Lists files with detailed permission information.
```bash
lshp
# Shows files with permissions, ownership, and sizes
# -rw-r--r-- user group 1024 file.txt
# drwxr-xr-x user group  256 folder/
```

### 20. see_path - Path Environment Viewer
Displays PATH variable in readable format.
```bash
see_path
# Output shows each PATH entry:
# /usr/local/bin
# /usr/bin
# /bin
# /usr/sbin
```

### 21. repl - Text Replacement
Advanced text replacement with regex support.
```bash
repl "old_text" "new_text" *.txt
# Replaces text in all .txt files

echo "hello world" | repl "world" "universe"
# Output: hello universe
```

### 22. tt - Text Tools
Various text processing utilities.
```bash
tt count file.txt
# Shows character, word, and line counts

tt format input.txt
# Applies text formatting
```

### 23. torel - Convert to Relative Path
Converts absolute paths to relative paths.
```bash
echo "/home/user/project/file.txt" | torel
# Output: ./file.txt (relative to current directory)

echo "/home/user/project/file.txt" | torel /home/user
# Output: ./project/file.txt (relative to specified base)

torel filelist.txt
# Converts all paths in file to relative paths
```

### 24. lsd - List Drives/Directories
Lists available drives and network mappings on Windows.
```bash
lsd
# Output shows available drives:
# C:
# D:
# E:
#
# Z: => \\server\share
```

### 25. ls_big_fn - List Large Files
Lists files larger than specified size threshold.
```bash
ls_big_fn 100M
# Shows files larger than 100MB

ls_big_fn 1G /path/to/check
# Shows files larger than 1GB in specific path
```

### 26. ecd - Enhanced Change Directory
Enhanced directory changing with validation.
```bash
ecd /home/user/projects
# Changes directory with error checking

ecd ..
# Goes up one directory level
```

### 27. ecds - Enhanced Change Directory with Stack
Directory navigation with stack management.
```bash
ecds
# Shows directory stack

ecds /new/path
# Changes directory and updates stack
```

### 28. !.bat - Quick Batch Execution
Executes batch commands in current context.
```bash
!.bat
# Executes predefined batch operations

!.bat custom_command
# Runs custom batch command
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/tlqtangok/perl_p.git
cd perl_p
```

2. Set environment variables:
```bash
# Linux/Unix
export perl_p=/path/to/perl_p
export PATH=$PATH:$perl_p

# Windows
set perl_p=C:\path\to\perl_p
set PATH=%PATH%;%perl_p%
```

3. Ensure dependencies are installed:
   - Perl 5.x with standard modules
   - Python 3.x
   - Git (for git-related tools)

## Configuration

The project includes configuration files:
- `.bashrc` - Linux/Unix shell aliases and functions
- `win_bin/` - Windows-specific executables
- Environment variable `perl_p` should point to installation directory

## Requirements

- **Perl**: Core scripting engine
- **Python**: Advanced processing capabilities
- **Git**: Version control integration
- **Standard Unix/Linux utilities**: grep, ls, tar, etc.
- **Windows**: cmd.exe and batch file support

## Contributing

This project welcomes contributions. Please ensure:
- Follow existing code style conventions
- Test utilities on target platforms
- Document new features with examples

## License

Open source project for enhancing command-line productivity.

---

**Note:** These utilities are designed for cross-platform development environments, with specific optimizations for both Unix/Linux and Windows systems.

