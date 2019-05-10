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
#pragma OPENCL EXTENSION cl_altera_channels : enable

#define BLOCK_SIZE 6
#define PB_BLOCK_SIZE (BLOCK_SIZE+1)
#define BLOCK_NUM_ELE BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE
#define sr_len (PB_BLOCK_SIZE * PB_BLOCK_SIZE + PB_BLOCK_SIZE + 2)

channel int REG_IDX __attribute__((depth(32)));
channel int LORENZO_IDX __attribute__((depth(32)));

channel float REG_DATA __attribute__((depth(32)));
channel float LORENZO_DATA __attribute__((depth(32)));

__attribute__((task))
__kernel void load_data(int r1, int r2, int r3, 
			__global float *restrict oriData,
			__global unsigned char *restrict indicator)
{
	// calculate block dims
	int num_x = (r1 - 1) / BLOCK_SIZE + 1;
	int num_y = (r2 - 1) / BLOCK_SIZE + 1;
	int num_z = (r3 - 1) / BLOCK_SIZE + 1;
	int num_blocks = num_x * num_y * num_z;
	int num_elements = r1 * r2 * r3;
	int dim0_offset = r2 * r3;
	int dim1_offset = r3;
	int i = 0, j = 0, k = 0;

	for(int n = 0; n < num_blocks; n++) {
		int data_pos = i * BLOCK_SIZE * dim0_offset + j * BLOCK_SIZE * dim1_offset + k * BLOCK_SIZE;
		size_t ii = 0, jj = 0, kk = 0;
		int x = 0, y = 0, z = 0;
		unsigned char indicator_n = indicator[n];
		if(!indicator_n)
			write_channel_altera(REG_IDX, n);
		else
			write_channel_altera(LORENZO_IDX, n);
		//#pragma unroll BLOCK_SIZE
		for(int m = 0; m < BLOCK_NUM_ELE; m++) {
			int pos = data_pos + x * dim0_offset + y * dim1_offset + z;
			z = (k * BLOCK_SIZE + kk + 1 < r3) ? z + 1 : z;
			float data = oriData[pos];
			// regression
			if(!indicator_n)
				write_channel_altera(REG_DATA, data);
			else
				write_channel_altera(LORENZO_DATA, data);
			kk += 1;
			if(kk == BLOCK_SIZE) {
				z = 0;
				y = (j * BLOCK_SIZE + jj + 1 < r2) ? y + 1 : y;
				jj += 1;
				kk = 0;
			}
			if(jj == BLOCK_SIZE) {
				z = 0;
				y = 0;
				x = (i * BLOCK_SIZE + ii + 1 < r1) ? x + 1 : x;
				ii += 1;
				jj = 0;
				kk = 0;
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
	}
}

__kernel void reg_pq(
		int num_reg,
		int intvCapacity, int intvRadius,
		double realPrecision,
		__global float4 *restrict reg_params,
		__global float *restrict unpredictable_data,   //output
		__global int *restrict blockwise_unpred_count, //output
		__global int *restrict type)                   //output
{
	for(int i = 0; i < num_reg; i++) {
		int n = read_channel_altera(REG_IDX);
		float4 reg_params_n = reg_params[n];
		int ii = 0, jj = 0, kk = 0;
		int block_unpredictable_count = 0;
		int type_pos[BLOCK_NUM_ELE];
		float unpred_data_pos[BLOCK_NUM_ELE];
		for(int m = 0; m < BLOCK_NUM_ELE; m++)
			unpred_data_pos[m] = 0;
		//#pragma unroll BLOCK_SIZE	
		for(int m = 0; m < BLOCK_NUM_ELE; m++) {
			float curData = read_channel_altera(REG_DATA);
			float pred = reg_params_n.s0 * ii + reg_params_n.s1 * jj + reg_params_n.s2 * kk + reg_params_n.s3;									
			double diff = curData - pred;
			double itvNum = fabs((float)diff)/realPrecision + 1;
			int stdIntvCap = intvCapacity;
			double signed_itvNum = (diff < 0) ? -itvNum : itvNum;
			int type_temp = (int) (signed_itvNum/2) + intvRadius;
			pred = pred + 2 * (type_temp - intvRadius) * realPrecision;
			if (itvNum >= stdIntvCap || fabs(curData - pred) > realPrecision) {
				type_pos[m] = 0;
				unpred_data_pos[block_unpredictable_count ++] = curData;
			} else {
				type_pos[m] = type_temp;
			}
			kk += 1;
			if(kk == BLOCK_SIZE) {
				jj += 1;
				kk = 0;
			}
			if(jj == BLOCK_SIZE) {
				ii += 1;
				jj = 0;
				kk = 0;
			}
		}
		blockwise_unpred_count[n] = block_unpredictable_count;
		for(int m = 0; m < BLOCK_NUM_ELE; m++)
			type[n * BLOCK_NUM_ELE + m]= type_pos[m];
		for(int m = 0; m < BLOCK_NUM_ELE; m++)
			unpredictable_data[n * BLOCK_NUM_ELE + m]= unpred_data_pos[m];
	}
}

__kernel void lorenzo_pq(
			int num_lorenzo,
			int intvCapacity_sz, int intvRadius,
			double realPrecision,
			__global float *restrict unpredictable_data,   //output
			__global int *restrict blockwise_unpred_count, //output
			__global int *restrict type)                   //output
{
	float realPrecision_inv = 1/realPrecision;
	float realPrecision_2 = realPrecision * 2;
	for(int i = 0; i < num_lorenzo; i++) {
		int n = read_channel_altera(LORENZO_IDX);
		int ii = 0, jj = 0, kk = 0;
		int block_unpredictable_count = 0;
		float shift_reg[sr_len];
		#pragma unroll
		for(int m = 0; m < sr_len; m++)
			shift_reg[m] = 0;
		int type_pos[BLOCK_NUM_ELE];
		float unpred_data_pos[BLOCK_NUM_ELE];
		for(int m = 0; m < BLOCK_NUM_ELE; m++)
			unpred_data_pos[m] = 0;
		//#pragma unroll BLOCK_SIZE
		for(int m = 0; m < BLOCK_NUM_ELE; m++) {
			float curData = read_channel_altera(LORENZO_DATA);
			float pred = shift_reg[sr_len - 2] + shift_reg[PB_BLOCK_SIZE * PB_BLOCK_SIZE + 1] + shift_reg[PB_BLOCK_SIZE + 1] 
						- shift_reg[PB_BLOCK_SIZE * PB_BLOCK_SIZE] - shift_reg[PB_BLOCK_SIZE] - shift_reg[1] + shift_reg[0];				
			float diff = curData - pred;
			float itvNum = mad(fabs(diff), realPrecision_inv, 1);
			int stdIntvCap = intvCapacity_sz;
			float signed_itvNum = (diff < 0) ? -itvNum : itvNum;
			int type_temp = (int) (signed_itvNum*0.5);
			pred = mad((float)type_temp, realPrecision_2, pred);
			//pred = diff + realPrecision_f + pred;
  			
			shift_reg[sr_len-1] = (itvNum >= stdIntvCap || fabs(curData - pred) > realPrecision) ? curData : pred;
			if (itvNum >= stdIntvCap || fabs(curData - pred) > realPrecision) {
				type_pos[m] = 0;
				unpred_data_pos[block_unpredictable_count ++] = curData;
			} else {
				type_pos[m] = type_temp + intvRadius;
			}
			/*
			#ifdef USE_MEAN
			type_pos[m] = (fabs(curData - mean) <= realPrecision) ? 1 : type_pos[m];
			shift_reg[sr_len-1] = (fabs(curData - mean) <= realPrecision) ? mean : shift_reg[sr_len-1];
			#endif
			*/

			// shift data
			#pragma unroll
			for(int m = 0; m < sr_len-1; m++)
				shift_reg[m] = shift_reg[m+1];
			shift_reg[sr_len - 1] = 0;
			kk += 1;
			if(kk == BLOCK_SIZE) {
				jj += 1;
				kk = 0;
				
				// shift data
				#pragma unroll
				for(int m = 1; m < sr_len; m++)
					shift_reg[m - 1] = shift_reg[m];
				shift_reg[sr_len - 1] = 0;
				
			}
			if(jj == BLOCK_SIZE) {
				ii += 1;
				jj = 0;
				kk = 0;
				
				// shift data
				#pragma unroll
				for(int m = PB_BLOCK_SIZE; m < sr_len; m++)
					shift_reg[m - PB_BLOCK_SIZE] = shift_reg[m];
				#pragma unroll
				for(int m = 0; m < PB_BLOCK_SIZE; m++)
					shift_reg[sr_len - m - 1] = 0;
				
			}
		}
		blockwise_unpred_count[n] = block_unpredictable_count;
		for(int m = 0; m < BLOCK_NUM_ELE; m++)
			type[n * BLOCK_NUM_ELE + m]= type_pos[m];
		for(int m = 0; m < BLOCK_NUM_ELE; m++)
			unpredictable_data[n * BLOCK_NUM_ELE + m]= unpred_data_pos[m];
	}
}
