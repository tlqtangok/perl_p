#pragma once 
#include <iostream>
#include <string>
using namespace std; 

namespace glo
{
	extern int ei;  // static can initialize here
	static int si = 5;

	extern string cn;

};

class com
{
	public:
		int x; 
		int y; 
		void init(int x_,int y_);
		void print();
		void _10t();

};
