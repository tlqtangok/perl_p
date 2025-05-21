#include "dep.h"

#if 0
class dep
{
public:
	int x;
	void printx();

};
#endif 



void dep::printx()
{

	glo::si *= 2;
	glo::ei *= 2;

	cout << "si:"<<glo::si << endl;
	cout << "ei:"<<glo::ei << endl;
	cout << endl;
}
void dep::setcom(com *pcom_)
{

	pcom = pcom_;

}



void dep::resetglo()
{

	auto &ecom = *pcom;

	glo::si = glo::si * 11; 	

}


