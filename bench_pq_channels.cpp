//#include "clwrap.hpp"
//#include "bench_pq.hpp"
//#include <sys/time.h>
#include <getopt.h>

#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
using namespace aocl_utils;
using namespace std;

// SZ block size
#define BLOCK_SIZE 6
#define BLOCK_NUM_ELE BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE

// quartus version
int version = 16;
// CL binary name
const char *binary_prefix = "pred_quant_channels_unroll";
// The set of simultaneous kernels
enum KERNELS {
  K_DATA,
  K_REG,
  K_LORENZO,
  K_NUM_KERNELS
};
static const char *kernel_names[K_NUM_KERNELS] =
{
  "load_data",
  "reg_pq",
  "lorenzo_pq"
};

bool init();
void cleanup();
// ACL runtime configuration
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_command_queue queues[K_NUM_KERNELS];
static cl_kernel kernels[K_NUM_KERNELS];
static cl_program program = NULL;
static cl_int status = 0;

static void usage(void)
{
	puts("");
	printf("Usage: bench_pq [options]\n");
	printf("\n");
	printf("-h : print this msg\n");
	printf("-v version : kernel version number (default: 1)\n");
	printf("-m mode  : kernel execution model (0 for single task, 1 for NDRange, default: 1)\n");
	printf("-i input_file : the path to the input file\n");
	printf("-c comparison_file : the path to the comparison file for verifiction\n");
	puts("");
}

static void bench_pq(int kver, int ndr_mode, string input_fn, string comp_fn)
{
	char kfn[80];
	char infn[input_fn.size() + 1];
	char compfn[comp_fn.size() + 1];

	strcpy(infn, input_fn.c_str());
	strcpy(compfn, comp_fn.c_str());

	FILE * input = fopen(infn, "rb");
	if(!input) {
		perror("failed to open input file");
		exit(-1);
	}
	int r1, r2, r3;
	int intvCapacity, intvCapacity_sz;
	int intvRadius;
	float mean;
	double realPrecision;
	int num_reg, num_lorenzo;
	fread(&r1, sizeof(int), 1, input);
	fread(&r2, sizeof(int), 1, input);
	fread(&r3, sizeof(int), 1, input);
	fread(&intvCapacity, sizeof(int), 1, input);
	fread(&intvRadius, sizeof(int), 1, input);
	fread(&mean, sizeof(float), 1, input);
	fread(&realPrecision, sizeof(double), 1, input);
	fread(&num_reg, sizeof(int), 1, input);
	fread(&num_lorenzo, sizeof(int), 1, input);
	int num_x = (r1 - 1) / BLOCK_SIZE + 1;
	int num_y = (r2 - 1) / BLOCK_SIZE + 1;
	int num_z = (r3 - 1) / BLOCK_SIZE + 1;
	int num_blocks = num_x * num_y * num_z;
	intvCapacity_sz = intvCapacity - 2;
	printf("total num blocks: %d, num_reg: %d, num_lorenzo: %d\n", num_blocks, num_reg, num_lorenzo);
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
	
	cl_mem d_oriData, d_regParams, d_indicator, d_type, d_unpredData, d_unpredCnt;
	d_oriData = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes_oriData, NULL, &status);
  	checkError(status, "Failed to allocate input device buffer\n");
	d_regParams = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes_reg_params, NULL, &status);
        checkError(status, "Failed to allocate input device buffer\n");
	d_indicator = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes_indicator, NULL, &status);
        checkError(status, "Failed to allocate input device buffer\n");
	d_unpredData = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes_unpredictable_data, NULL, &status);
        checkError(status, "Failed to allocate output device buffer\n");
	d_unpredCnt = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes_blockwise_unpred_count, NULL, &status);
        checkError(status, "Failed to allocate output device buffer\n");
	d_type = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes_type, NULL, &status);
        checkError(status, "Failed to allocate output device buffer\n");

  	status = clEnqueueWriteBuffer(queues[K_DATA], d_oriData, CL_TRUE, 0, bytes_oriData, oriData, 0, NULL, NULL);
  	checkError(status, "Failed to copy data to device");
  	status = clEnqueueWriteBuffer(queues[K_DATA], d_indicator, CL_TRUE, 0, bytes_indicator, indicator, 0, NULL, NULL);
  	checkError(status, "Failed to copy data to device");
  	status = clEnqueueWriteBuffer(queues[K_REG], d_regParams, CL_TRUE, 0, bytes_reg_params, reg_params, 0, NULL, NULL);
  	checkError(status, "Failed to copy data to device");

 	status = clSetKernelArg(kernels[K_DATA], 0, sizeof(cl_int), (void*)&r1);
  	checkError(status, "Failed to set kernel_writer arg 0");
 	status = clSetKernelArg(kernels[K_DATA], 1, sizeof(cl_int), (void*)&r2);
  	checkError(status, "Failed to set kernel_writer arg 0");
 	status = clSetKernelArg(kernels[K_DATA], 2, sizeof(cl_int), (void*)&r3);
  	checkError(status, "Failed to set kernel_writer arg 0");
 	status = clSetKernelArg(kernels[K_DATA], 3, sizeof(cl_mem), (void*)&d_oriData);
  	checkError(status, "Failed to set kernel_writer arg 0");
 	status = clSetKernelArg(kernels[K_DATA], 4, sizeof(cl_mem), (void*)&d_indicator);
  	checkError(status, "Failed to set kernel_writer arg 0");

 	status = clSetKernelArg(kernels[K_REG], 0, sizeof(cl_int), (void*)&num_reg);
  	checkError(status, "Failed to set kernel_writer arg 0");
 	status = clSetKernelArg(kernels[K_REG], 1, sizeof(cl_int), (void*)&intvCapacity);
  	checkError(status, "Failed to set kernel_writer arg 0");
 	status = clSetKernelArg(kernels[K_REG], 2, sizeof(cl_int), (void*)&intvRadius);
  	checkError(status, "Failed to set kernel_writer arg 0");
 	status = clSetKernelArg(kernels[K_REG], 3, sizeof(cl_double), (void*)&realPrecision);
  	checkError(status, "Failed to set kernel_writer arg 0");
 	status = clSetKernelArg(kernels[K_REG], 4, sizeof(cl_mem), (void*)&d_regParams);
  	checkError(status, "Failed to set kernel_writer arg 0");
 	status = clSetKernelArg(kernels[K_REG], 5, sizeof(cl_mem), (void*)&d_unpredData);
  	checkError(status, "Failed to set kernel_writer arg 0");
 	status = clSetKernelArg(kernels[K_REG], 6, sizeof(cl_mem), (void*)&d_unpredCnt);
  	checkError(status, "Failed to set kernel_writer arg 0");
 	status = clSetKernelArg(kernels[K_REG], 7, sizeof(cl_mem), (void*)&d_type);
  	checkError(status, "Failed to set kernel_writer arg 0");
	
 	status = clSetKernelArg(kernels[K_LORENZO], 0, sizeof(cl_int), (void*)&num_lorenzo);
  	checkError(status, "Failed to set kernel_writer arg 0");
 	status = clSetKernelArg(kernels[K_LORENZO], 1, sizeof(cl_int), (void*)&intvCapacity_sz);
  	checkError(status, "Failed to set kernel_writer arg 0");
 	status = clSetKernelArg(kernels[K_LORENZO], 2, sizeof(cl_int), (void*)&intvRadius);
  	checkError(status, "Failed to set kernel_writer arg 0");
 	status = clSetKernelArg(kernels[K_LORENZO], 3, sizeof(cl_double), (void*)&realPrecision);
  	checkError(status, "Failed to set kernel_writer arg 0");
 	status = clSetKernelArg(kernels[K_LORENZO], 4, sizeof(cl_mem), (void*)&d_unpredData);
  	checkError(status, "Failed to set kernel_writer arg 0");
 	status = clSetKernelArg(kernels[K_LORENZO], 5, sizeof(cl_mem), (void*)&d_unpredCnt);
  	checkError(status, "Failed to set kernel_writer arg 0");
 	status = clSetKernelArg(kernels[K_LORENZO], 6, sizeof(cl_mem), (void*)&d_type);
  	checkError(status, "Failed to set kernel_writer arg 0");

	if(ndr_mode == 1) {
#ifdef ENABLE_INTELFPGA
	snprintf(kfn, sizeof(kfn), "pred_quant_ndr_v%d.aocx", kver);
#else
	snprintf(kfn, sizeof(kfn), "pred_quant_ndr_v%d", kver);
#endif
	} else {
#ifdef ENABLE_INTELFPGA
	snprintf(kfn, sizeof(kfn), "pred_quant_channels_6.aocx");
#else
	snprintf(kfn, sizeof(kfn), "pred_quant_v%d", kver);
#endif
	}
	printf("kernel: %s\n", kfn);

	printf("Execution Start\n");
	double time = getCurrentTimestamp();
	cl_event kernel_event;
	status = clEnqueueTask(queues[K_DATA], kernels[K_DATA], 0, NULL, &kernel_event);
	checkError(status, "Failed to launch kernel_writer");
	status = clEnqueueTask(queues[K_REG], kernels[K_REG], 0, NULL, NULL);
	checkError(status, "Failed to launch kernel_search");
	status = clEnqueueTask(queues[K_LORENZO], kernels[K_LORENZO], 0, NULL, NULL);
	checkError(status, "Failed to launch kernel_search");
	for(int i=0; i<K_NUM_KERNELS; ++i) {
	status = clFinish(queues[i]);
	checkError(status, "Failed to finish (%d: %s)", i, kernel_names[i]);
	}

	// Record execution time
	time = getCurrentTimestamp() - time;
	printf("Kernel execution time = %.4fms\n", time * 1E3);

	// Copy results from device to host
	status = clEnqueueReadBuffer(queues[K_REG], d_unpredData, CL_TRUE, 0, bytes_unpredictable_data, unpredictable_data, 0, NULL, NULL);
	checkError(status, "Failed to copy data from device");
	status = clEnqueueReadBuffer(queues[K_REG], d_unpredCnt, CL_TRUE, 0, bytes_blockwise_unpred_count, blockwise_unpred_count, 0, NULL, NULL);
	checkError(status, "Failed to copy data from device");
	status = clEnqueueReadBuffer(queues[K_REG], d_type, CL_TRUE, 0, bytes_type, type, 0, NULL, NULL);
	checkError(status, "Failed to copy data from device");


	// verification
	printf("Verification Start\n");
	FILE * output = fopen(compfn, "rb");
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
	fclose(output);

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
			//printf("%04d: %08x %08x\n", i, type[i], ref_type[i]);
			if (type[i] != ref_type[i]) break;
		}
		success = false;
	}

	if(success)
		printf("Verification Success\n");
	else
		printf("Verification Fail\n");

	
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
	
}

int main(int argc, char *argv[])
{
	int kver = 1;
	int ndr_mode = 1;
	string input_file = "../testdata/input_testfloat_8_8_128.dat";
	string comparison_file = "../testdata/output_testfloat_8_8_128.dat";

	while (1) {
                int option_index = 0;
                int opt;
                static struct option long_options[] = {
                        {"help",  no_argument,       0,  0 },
                        {0,        0,                 0,  0 }
                };

                opt = getopt_long(argc, argv, "hs:v:m:i:c:",
                                  long_options, &option_index);
                if (opt == -1)
                        break;

                switch (opt) {
                case 0: /* long opt */
                        if (strcmp(long_options[option_index].name, "shared") == 0) {
                                usage();
                                exit(1);
                        }
                        break;
                case 'h':
                        usage();
                        exit(1);
                case 'v':
                        kver = atoi(optarg);
                        break;
                case 'm':
                        ndr_mode = atoi(optarg);
			if(ndr_mode != 0 && ndr_mode != 1) {
				printf("-m could only be either 0 or 1\n");
				exit(1);
			}
                        break;
		case 'i':
			input_file = string(optarg);
			break;
		case 'c':
			comparison_file = string(optarg);
			break;
                default:
                        printf("Unknown option: %c\n", opt);
                        usage();
                        exit(1);
                }
        }
	printf("[bench_pq]\n");
	string mode_str[2] = {"Single Task", "NDRange"};
	printf("MODE=%d %s\n", ndr_mode, mode_str[ndr_mode].c_str());
	printf("VERSION=%d\n", kver);
	printf("INPUT_FILE=%s\n", input_file.c_str());
	printf("COMPARISON_FILE=%s\n", comparison_file.c_str());

  	if (!init())
    		return false;
  	printf("Init complete!\n");

	bench_pq(kver, ndr_mode, input_file, comparison_file);

	return 0;
}

// Set up the context, device, kernels, and buffers...
bool init() {
  cl_int status;

  // Start everything at NULL to help identify errors
  for(int i = 0; i < K_NUM_KERNELS; ++i){
    kernels[i] = NULL;
    queues[i] = NULL;
  }

  // Locate files via. relative paths
  if(!setCwdToExeDir())
    return false;

  // Get the OpenCL platform.
  if(version == 16)
    platform = findPlatform("Altera");
  else
    platform = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
  if(platform == NULL) {
    printf("ERROR: Unable to find OpenCL platform\n");
    return false;
  }

  // Query the available OpenCL devices and just use the first device if we find
  // more than one
  scoped_array<cl_device_id> devices;
  cl_uint num_devices;
  devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
  device = devices[0];

  // Create the context.
  context = clCreateContext(NULL, 1, &device, &oclContextCallback, NULL, &status);
  checkError(status, "Failed to create context");

  // Create the command queues
  for(int i=0; i<K_NUM_KERNELS; ++i) {
    queues[i] = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue (%d)", i);
  }

  // Create the program.
  std::string binary_file = getBoardBinaryFile(binary_prefix, device);
  printf("Using AOCX: %s\n\n", binary_file.c_str());
  program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  // Create the kernel - name passed in here must match kernel name in the
  // original CL file, that was compiled into an AOCX file using the AOC tool
  for(int i=0; i<K_NUM_KERNELS; ++i) {
    kernels[i] = clCreateKernel(program, kernel_names[i], &status);
    checkError(status, "Failed to create kernel (%d: %s)", i, kernel_names[i]);
  }

  return true;
}

// Free the resources allocated during initialization
void cleanup() {
  for(int i=0; i<K_NUM_KERNELS; ++i)
    if(kernels[i]) 
      clReleaseKernel(kernels[i]);  
  if(program) 
    clReleaseProgram(program);
  for(int i=0; i<K_NUM_KERNELS; ++i)
    if(queues[i]) 
      clReleaseCommandQueue(queues[i]);
  if(context) 
    clReleaseContext(context);
}
