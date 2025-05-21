
#include "dep1.h"




void dep1::printx()
{
	//cout << x << endl;

	glo::si *= 2;
	glo::ei *= 2;
	cout << "si:"<<glo::si << endl;
	cout << "ei:"<<glo::ei << endl;
	cout << endl;
}


void dep1::setcom(com *pcom_)
{

	pcom = pcom_;

}

void dep1::resetglo()
{

	auto &ecom = *pcom;

	glo::si = glo::si * 10; 	

}
