#include "clwrap.hpp"
#include <sys/time.h>

// A sample code for simpleOCLInit.hpp
// Written by Kaz Yoshii <ky@anl.gov>

static void bench_pq()
{
	clWrap  cw;

	int r1, r2, r3;
	int intvCapacity;
	int intvRadius;
	float mean;
	double realPrecision;
	float * oriData;
	float * reg_params;
	unsigned char * indicator;
	float * unpredictable_data;
	int * blockwise_unpred_count;
	int * type;
	size_t bytes_oriData = 0xbad;
	size_t bytes_reg_params = 0xbad;
	size_t bytes_indicator = 0xbad;
	size_t bytes_unpredictable_data = 0xbad;
	size_t bytes_blockwise_unpred_count = 0xbad;
	size_t bytes_type = 0xbad;

	/* XXX: test params and data needs to be loaded
	   assigning dummy numbers/data for now
	 */
	r1 = r2 = r3 = 16;
	intvCapacity = intvRadius = 1;
	mean = 1.0;
	realPrecision = 1e-4;

	oriData = (float *)malloc(bytes_oriData);
	reg_params = (float *)malloc(bytes_reg_params);
	indicator =  (unsigned char *)malloc(bytes_indicator);
	unpredictable_data = (float *)malloc(bytes_unpredictable_data);
	blockwise_unpred_count = (int *)malloc(bytes_blockwise_unpred_count);
	type = (int *)malloc(bytes_type);
	
	cw.listPlatforms();
	cw.listDevices();

	cw.prepKernel("pred_quant.aocx", "pred_and_quant");

#if 0
__kernel void pred_and_quant(int r1, int r2, int r3, 
			int intvCapacity, int intvRadius,
			float mean,
			double realPrecision,
			__global float *restrict oriData,
			__global float *restrict reg_params,
			__global unsigned char *restrict indicator,
			__global float *restrict unpredictable_data,   //output
			__global int *restrict blockwise_unpred_count, //output
			__global int *restrict type)                   //output
#endif

	cw.appendArg(sizeof(cl_int), &r1);
	cw.appendArg(sizeof(cl_int), &r2);
	cw.appendArg(sizeof(cl_int), &r3);
	cw.appendArg(sizeof(cl_int), &intvCapacity);
	cw.appendArg(sizeof(cl_int), &intvRadius);
	cw.appendArg(sizeof(cl_float), &mean);
	cw.appendArg(sizeof(cl_double), &realPrecision);
	

	cw.appendArg(bytes_oriData, oriData, cw.HOST2DEV);
	cw.appendArg(bytes_reg_params, reg_params, cw.HOST2DEV);
	cw.appendArg(bytes_indicator, indicator, cw.HOST2DEV);
	cw.appendArg(bytes_unpredictable_data, unpredictable_data, cw.DEV2HOST);
	cw.appendArg(bytes_blockwise_unpred_count, blockwise_unpred_count, cw.DEV2HOST);
	cw.appendArg(bytes_type, type,  cw.DEV2HOST);

	cw.runKernel(1);
};

int main(int argc, char *argv[])
{
	bench_pq();

	return 0;
}
