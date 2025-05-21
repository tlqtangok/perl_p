#include <iostream>
using namespace std; 

#include "com.h"

#if 1 // init
//int glo::si = 777;

int glo::ei = 55;
string glo::cn = "吕布";
#endif 

void com::init(int x_, int y_)
{
	x = x_; 
	y = y_;

	glo::cn = glo::cn + glo::cn;

	//glo::si = x*y; 
}

void com::print()
{
	cout << glo::si << endl;
}

void com::_10t()
{
	glo::si = glo::si * 10;
}
