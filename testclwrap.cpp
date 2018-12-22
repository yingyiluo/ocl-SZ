#include "clwrap.hpp"
#include <sys/time.h>

// A sample code for simpleOCLInit.hpp
// Written by Kaz Yoshii <ky@anl.gov>

// source $OPENCLENV
// g++ -I. -Wall -O2 -g -Wno-unknown-pragmas `aocl compile-config` -std=c++11 -o testclwrap testclwrap.cpp `aocl link-config`
// aocl

static void test_clwrap()
{
	clWrap  cw;

	cw.listPlatforms();
	cw.listDevices();

	int gsiz = 8;
	int lsiz = 2;

	int *a0 = new int[gsiz];
	int *a1 = new int[gsiz];

	cw.prepKernel("dummy.aocx", "dummy");

	cw.appendArg(sizeof(int)*gsiz, a0, cw.DEV2HOST);
	cw.appendArg(sizeof(int)*gsiz, a1, cw.DEV2HOST);

	cw.runKernel(gsiz, lsiz);

	for (int i = 0; i < gsiz; i++) 
		cout << i << "," << a0[i] << "," << a1[i] << " ";
	cout << endl;
};

int main()
{
	test_clwrap();

	return 0;
}
