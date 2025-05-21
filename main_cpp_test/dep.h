#pragma once

#include <iostream>

#include "com.h"
using namespace std; 

class dep
{
public:
	int x;
	void printx();
	void setcom(com *pcom_);
	com *pcom; 

	void resetglo();

};
