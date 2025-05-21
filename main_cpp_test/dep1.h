#pragma once

#include <iostream>

#include "com.h"

using namespace std; 

class dep1
{
public:
	int x;
	void printx();
	void setcom(com *pcom_);
	void resetglo();
	com *pcom;
};
