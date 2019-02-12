/* 
* Prediction & Quantization 
* This kernel is for regression + mean/classic Lorenzo algorithm.
*/

// int * blockwise_unpred_count = (int *) malloc(num_blocks * sizeof(int));
// int * blockwise_unpred_count_pos = blockwise_unpred_count;
// int * result_type = (int *) malloc(num_blocks*MAX_NUM_ELE * sizeof(int));
// float * reg_params = (float *) malloc(num_blocks * 4 * sizeof(float));
// float * reg_params_pos = reg_params;
// unsigned char * indicator = (unsigned char *) malloc(num_blocks * sizeof(unsigned char));
// memset(indicator, 0, num_blocks * sizeof(unsigned char));
// unsigned char * indicator_pos = indicator;
// 	size_t unpred_data_max_size = max_num_block_elements;
//	float * result_unpredictable_data = (float *) malloc(unpred_data_max_size * sizeof(float) * num_blocks);
// LYY: can convert uchar to bit vector to save space and make it constant
// LYY: may also want to pass realPrecision's reciprocal to kernel
// LYY: if(k * BLOCK_SIZE + kk + 1 < r3) block_data_pos_z ++;  //boundary checking, we can do padding

#define BLOCK_SIZE 6
#define PB_BLOCK_SIZE (BLOCK_SIZE+1)
#define BLOCK_NUM_ELE BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE

__attribute__((task))
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
{
	// calculate block dims
	int num_x = (r1 - 1) / BLOCK_SIZE + 1;
	int num_y = (r2 - 1) / BLOCK_SIZE + 1;
	int num_z = (r3 - 1) / BLOCK_SIZE + 1;
	int num_blocks = num_x * num_y * num_z;
	int num_elements = r1 * r2 * r3;
	int dim0_offset = r2 * r3;
	int dim1_offset = r3;
	// move regression part out
	int params_offset_b = num_blocks;
	int params_offset_c = 2 * num_blocks;
	int params_offset_d = 3 * num_blocks;
	// for sz method
	int strip_dim0_offset = PB_BLOCK_SIZE * PB_BLOCK_SIZE;
	int strip_dim1_offset = PB_BLOCK_SIZE;
	// invtCapacity is from external param: exe_params
	int intvCapacity_sz = intvCapacity - 2;
	int total_unpred = 0;
	int reg_params_pos = 0;
	int i = 0, j = 0, k = 0;
	__local float pred_buffer_pos[PB_BLOCK_SIZE * PB_BLOCK_SIZE * PB_BLOCK_SIZE];
	__local int type_pos[BLOCK_NUM_ELE];
	for(int n = 0; n < num_blocks; n++) {
		int data_pos = i * BLOCK_SIZE * dim0_offset + j * BLOCK_SIZE * dim1_offset + k * BLOCK_SIZE;
		for(int m = 0; m < PB_BLOCK_SIZE * PB_BLOCK_SIZE * PB_BLOCK_SIZE; m++)
			pred_buffer_pos[m] = 0;
		int idx = PB_BLOCK_SIZE * PB_BLOCK_SIZE + PB_BLOCK_SIZE + 1;
		int ii = 0, jj = 0, kk = 0;
		int x = 0, y = 0, z = 0;
		for(int m = 0; m < BLOCK_NUM_ELE; m++) {
			int pos = data_pos + x * dim0_offset + y * dim1_offset + z;
			z = (k * BLOCK_SIZE + kk + 1 < r3) ? z + 1 : z;
			pred_buffer_pos[idx] = oriData[pos];
			idx += 1;
			kk += 1; 
			if(kk == BLOCK_SIZE) {
				z = 0;
				y = (j * BLOCK_SIZE + jj + 1 < r2) ? y + 1 : y;
				jj += 1;
				kk = 0;
				idx += 1;
			}
			if(jj == BLOCK_SIZE) {
				z = 0;
				y = 0;
				x = (i * BLOCK_SIZE + ii + 1 < r1) ? x + 1 : x;
				ii += 1;
				jj = 0;
				kk = 0;
				idx += PB_BLOCK_SIZE;
			}
		}

		k += 1;
		if(k == num_z) {
			j += 1;
			k = 0;
		}
		if(j == num_y) {
			i += 1;
			j = 0;
			k = 0;
		}

		// When !(indicator[k]), use linear regression 
		ii = 0, jj = 0, kk = 0;
		idx = PB_BLOCK_SIZE * PB_BLOCK_SIZE + PB_BLOCK_SIZE + 1;
		int block_unpredictable_count = 0;
		unsigned char indicator_n = indicator[n];
		for(int m = 0; m < BLOCK_NUM_ELE; m++) {
			float curData = pred_buffer_pos[idx];
			float pred = indicator_n ? 
						pred_buffer_pos[idx - 1] + pred_buffer_pos[idx - strip_dim1_offset] + pred_buffer_pos[idx - strip_dim0_offset] - pred_buffer_pos[idx - strip_dim1_offset - 1]
						- pred_buffer_pos[idx - strip_dim0_offset - 1] - pred_buffer_pos[idx - strip_dim0_offset - strip_dim1_offset] + pred_buffer_pos[idx - strip_dim0_offset - strip_dim1_offset - 1] 
						: (reg_params[reg_params_pos] * ii + reg_params[reg_params_pos + params_offset_b] * jj + reg_params[reg_params_pos + params_offset_c] * kk + reg_params[reg_params_pos + params_offset_d]);									
			double diff = curData - pred;
			double itvNum = fabs((float)diff)/realPrecision + 1;
			int stdIntvCap = indicator_n ? intvCapacity_sz : intvCapacity;
			if (itvNum < stdIntvCap){
				if (diff < 0) itvNum = -itvNum;
				type_pos[m] = (int) (itvNum/2) + intvRadius;
				pred_buffer_pos[idx] = pred + 2 * (type_pos[m] - intvRadius) * realPrecision;
				//ganrantee comporession error against the case of machine-epsilon
				if(fabs(curData - pred_buffer_pos[idx]) > realPrecision){	
					type_pos[m] = 0;
					pred_buffer_pos[idx] = curData;
					unpredictable_data[total_unpred + block_unpredictable_count ++] = curData;
				}		
			}
			else{
				type_pos[m] = 0;
				pred_buffer_pos[idx] = curData;
				unpredictable_data[total_unpred + block_unpredictable_count ++] = curData;
			}
		
			#ifdef USE_MEAN
			type_pos[m] = indicator_n && (fabs(curData - mean) <= realPrecision) ? 1 : type_pos[m];
			pred_buffer_pos[idx] = indicator_n && (fabs(curData - mean) <= realPrecision) ? mean : pred_buffer_pos[idx];
			#endif

			idx += 1;
			kk += 1;
			if(kk == BLOCK_SIZE) {
				jj += 1;
				kk = 0;
				idx += 1;
			}
			if(jj == BLOCK_SIZE) {
				ii += 1;
				jj = 0;
				kk = 0;
				idx += PB_BLOCK_SIZE;
			}
		}
		reg_params_pos = indicator_n ? reg_params_pos : (reg_params_pos + 1);
		total_unpred += block_unpredictable_count;
		blockwise_unpred_count[n] = block_unpredictable_count;
		for(int m = 0; m < BLOCK_NUM_ELE; m++)
			type[n * BLOCK_NUM_ELE + m]= type_pos[m];
	}
}
