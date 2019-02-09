#include "clwrap.hpp"
#include <sys/time.h>

// A sample code for simpleOCLInit.hpp
// Written by Kaz Yoshii <ky@anl.gov>

// Minimum alignment requirement to ensure use of DMA
#define AOCL_ALIGNMENT 64

// SZ block size
#define BLOCK_SIZE 6
#define BLOCK_NUM_ELE BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE

// Helper function
void *alignedMalloc(size_t size) {
        void *result = NULL;
        posix_memalign(&result, AOCL_ALIGNMENT, size);
        return result;
}

static void bench_pq(int kver)
{
	clWrap  cw;
	FILE * input = fopen("testdata/input_testfloat_8_8_128.dat", "rb");
	char kfn[80];

	if(!input) {
		perror("failed to open input file");
		exit(-1);
	}
	int r1, r2, r3;
	int intvCapacity;
	int intvRadius;
	float mean;
	double realPrecision;
	fread(&r1, sizeof(int), 1, input);
	fread(&r2, sizeof(int), 1, input);
	fread(&r3, sizeof(int), 1, input);
	fread(&intvCapacity, sizeof(int), 1, input);
	fread(&intvRadius, sizeof(int), 1, input);
	fread(&mean, sizeof(float), 1, input);
	fread(&realPrecision, sizeof(double), 1, input);
	int num_x = (r1 - 1) / BLOCK_SIZE + 1;
	int num_y = (r2 - 1) / BLOCK_SIZE + 1;
	int num_z = (r3 - 1) / BLOCK_SIZE + 1;
	int num_blocks = num_x * num_y * num_z;

	float * oriData;
	float * reg_params;
	unsigned char * indicator;
	float * unpredictable_data;
	int * blockwise_unpred_count;
	int * type;
	size_t bytes_oriData = r1 * r2 * r3 * sizeof(float);
	size_t bytes_reg_params = 4 * num_blocks * sizeof(float);
	size_t bytes_indicator = num_blocks * sizeof(unsigned char);
	size_t bytes_unpredictable_data = BLOCK_NUM_ELE * num_blocks * sizeof(float);
	size_t bytes_blockwise_unpred_count = num_blocks * sizeof(int);
	size_t bytes_type = BLOCK_NUM_ELE * num_blocks * sizeof(int);

	// input arrays
	oriData = (float *)alignedMalloc(bytes_oriData);
	reg_params = (float *)alignedMalloc(bytes_reg_params);
	indicator =  (unsigned char *)alignedMalloc(bytes_indicator);
	fread(oriData, sizeof(float), bytes_oriData / sizeof(float), input);
	fread(reg_params, sizeof(float), bytes_reg_params / sizeof(float), input);
	fread(indicator, sizeof(unsigned char), bytes_indicator / sizeof(unsigned char), input);
	fclose(input);
	// output arrays
	unpredictable_data = (float *)alignedMalloc(bytes_unpredictable_data);
	blockwise_unpred_count = (int *)alignedMalloc(bytes_blockwise_unpred_count);
	type = (int *)alignedMalloc(bytes_type);
	
	cw.listPlatforms();
	cw.listDevices();

#ifdef ENABLE_INTELFPGA
	snprintf(kfn, sizeof(kfn), "pred_quant_v%d.aocx", kver);
#else
	snprintf(kfn, sizeof(kfn), "pred_quant_v%d", kver);
#endif
	printf("kernel: %s\n", kfn);
	cw.prepKernel(kfn, "pred_and_quant");

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

	printf("Execution Start\n");
	cw.runKernel(1);

	double runtime = cw.getKernelElapsedNanoSec();
	printf("Kernel execution time = %.4fms\n", runtime * 1E-6);

	// verification
	printf("Verification Start\n");
	FILE * output = fopen("testdata/output_testfloat_8_8_128.dat", "rb");
	if(!output) {
		perror("failed to open output file");
		exit(-1);
	}

	float * ref_unpredictable_data = (float *)malloc(bytes_unpredictable_data);
	int * ref_blockwise_unpred_count = (int *)malloc(bytes_blockwise_unpred_count);
	int * ref_type = (int *)malloc(bytes_type);
	fread(ref_unpredictable_data, sizeof(float), bytes_unpredictable_data / sizeof(float), output);
	fread(ref_blockwise_unpred_count, sizeof(int), bytes_blockwise_unpred_count / sizeof(int), output);
	fread(ref_type, sizeof(int), bytes_type / sizeof(int), output);
	bool success = true;
	if(memcmp(unpredictable_data, ref_unpredictable_data, bytes_unpredictable_data)) {
		printf("unpredictable_data unmatch\n");
		success = false;
	}
	if(memcmp(blockwise_unpred_count, ref_blockwise_unpred_count, bytes_blockwise_unpred_count)) {
		printf("blockwise_unpred_count unmatch\n");
		success = false;
	}
	if(memcmp(type, ref_type, bytes_type)) {
		printf("type unmatch\n");
		for(int i=0; i<(int)(bytes_type/sizeof(int)); i++) {
			printf("%04d: %08x %08x\n", i, type[i], ref_type[i]);
			if (type[i] != ref_type[i]) break;
		}
		success = false;
	}

	if(success)
		printf("Verification Success\n");
	else
		printf("Verification Fail\n");

	/*
	// debug
 	for(int i = 0; i < BLOCK_NUM_ELE * num_blocks; i++) {
		if(unpredictable_data[i] != ref_unpredictable_data[i])
			printf("i: %d, %f, %f\n", i, unpredictable_data[i], ref_unpredictable_data[i]);
		if(type[i] != ref_type[i])
			printf("i: %d, %d, %d\n", i, type[i], ref_type[i]);

	}
	for(int i = 0; i < num_blocks; i++) {
		if(blockwise_unpred_count[i] != ref_blockwise_unpred_count[i])
			printf("i: %d, %d, %d\n", i, blockwise_unpred_count[i], ref_blockwise_unpred_count[i]);
	}
	*/
};

int main(int argc, char *argv[])
{
	int kver = 2;

	if (argc >= 2) {
		kver = atoi(argv[1]);
	}

	if (kver < 1 || kver > 2) {
		printf("kver should be 1 or 2\n");
		return 1;
	}

	bench_pq(kver);

	return 0;
}
