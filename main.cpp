#include <iostream>
#include "fastccd.h"
using namespace std;

int main(int argc, char *argv[])
{

    FastCCD inst;

    inst.parseParameter(argc, argv);
    inst.dumpParameter();
    inst.run();

    cout<<"hello"<<endl;

    return 0;
}
