//#include "common.hpp"
/*
* jd create at 2018-02-28
* to build & run on Linux, just run :
* make
*/
//#pragma once
#pragma warning(disable:4996)
#pragma warning(disable:4018)

#include <iostream>
#include <fstream>

#include <string>
#include <sstream>
#include <vector>
#include <stdint.h>  // uint64_t
#include <assert.h>

#include <vector>
#include <sstream>
#include <map>
#include <algorithm>    // std::sort
#include <utility>

#include <stack>
#include <memory.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <thread>
//#include <mutex>

#if __linux__
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
//#include <htslib/faidx.h>
#include <strings.h>
#include <getopt.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>

#if 0
	#include "htslib/tbx.h"
	#include "htslib/sam.h"
	#include "htslib/vcf.h"
	#include "htslib/kseq.h"
	#include "htslib/bgzf.h"
	#include "htslib/hts.h"
	#include "htslib/regidx.h"
	#include "hfile_internal.h"
	#include <unordered_map>
#endif 

#include <iomanip>
#include <list>
#include <unistd.h>
#include <sys/time.h>
#include <numeric>
#include <float.h>


#endif
using namespace std; 

#ifdef __linux__
#define __P__  return 0;   //__
#else
#define __P__  system("pause");return 0;   //__
#define popen(fp, fr) _popen((fp),(fr))
#define pclose(fp) _pclose(fp)
#endif

#define DEL_ARR_MEM(P) if(NULL != (P)){delete [] (P); (P) = NULL;}

void td_exec(int &args)
{
	cout << args << endl; 
}
// st__
int main(int argc, char **argv)
{
	int int_args = 9; 
	auto td_0 = std::thread(td_exec, std::ref(int_args)); 
	td_0.join();
	
	cout << 1111 << endl ;
	return 0; 
}

