/* 
* Prediction & Quantization: baseline
*/

#define block_size 6
#define pred_buffer_block_size (block_size+1)

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
	int num_x = (r1 - 1) / block_size + 1;
	int num_y = (r2 - 1) / block_size + 1;
	int num_z = (r3 - 1) / block_size + 1;
	int num_blocks = num_x * num_y * num_z;
	int num_elements = r1 * r2 * r3;
	int dim0_offset = r2 * r3;
	int dim1_offset = r3;
	// move regression part out
	int params_offset_b = num_blocks;
	int params_offset_c = 2 * num_blocks;
	int params_offset_d = 3 * num_blocks;
	// for sz method
	int strip_dim0_offset = pred_buffer_block_size * pred_buffer_block_size;
	int strip_dim1_offset = pred_buffer_block_size;
	// invtCapacity is from external param: exe_params
	int intvCapacity_sz = intvCapacity - 2;
	int total_unpred = 0;
	__global float *data_pos = oriData;
	__global float *block_data_pos_x;
	__global float *block_data_pos_y;
	__global float *block_data_pos_z;
	float pred_buffer[pred_buffer_block_size * pred_buffer_block_size * pred_buffer_block_size] = {0.0f};
	float *pred_buffer_pos;

	for(size_t i=0; i<num_x; i++){
		for(size_t j=0; j<num_y; j++){
			for(size_t k=0; k<num_z; k++){
				data_pos = oriData + i*block_size * dim0_offset + j*block_size * dim1_offset + k*block_size;
				// add 1 in x, y, z offset
				pred_buffer_pos = pred_buffer + pred_buffer_block_size * pred_buffer_block_size + pred_buffer_block_size + 1;
				block_data_pos_x = data_pos;
				for(int ii=0; ii<block_size; ii++){
					block_data_pos_y = block_data_pos_x;
					for(int jj=0; jj<block_size; jj++){
						block_data_pos_z = block_data_pos_y;
						for(int kk=0; kk<block_size; kk++){
							*pred_buffer_pos = *block_data_pos_z;
							if(k*block_size + kk + 1< r3) block_data_pos_z ++;  //boundary checking, we can do padding
							pred_buffer_pos ++;
						}
						// add 1 in z offset
						pred_buffer_pos ++;
						if(j*block_size + jj + 1< r2) block_data_pos_y += dim1_offset; //boundary checking
					} 
					// add 1 in y offset
					pred_buffer_pos += pred_buffer_block_size;
					if(i*block_size + ii + 1< r1) block_data_pos_x += dim0_offset; //boundary checking
				}
				if(!(indicator[k])){ //use linear regression 
					float curData;
					float pred;
					double itvNum;
					double diff;
					size_t index = 0;
					size_t block_unpredictable_count = 0;
					float * cur_data_pos = pred_buffer + pred_buffer_block_size * pred_buffer_block_size + pred_buffer_block_size + 1;
					for(size_t ii=0; ii<block_size; ii++){   
						for(size_t jj=0; jj<block_size; jj++){
							for(size_t kk=0; kk<block_size; kk++){
								curData = *cur_data_pos;
								pred = reg_params[0] * ii + reg_params[params_offset_b] * jj + reg_params[params_offset_c] * kk + reg_params[params_offset_d];									
								diff = curData - pred;
								itvNum = fabs(diff)/realPrecision + 1;
								if (itvNum < intvCapacity){
									if (diff < 0) itvNum = -itvNum;
									type[index] = (int) (itvNum/2) + intvRadius;
									pred = pred + 2 * (type[index] - intvRadius) * realPrecision;
									//ganrantee comporession error against the case of machine-epsilon
									if(fabs(curData - pred)>realPrecision){	
										type[index] = 0;
										//pred = curData; // no use
										unpredictable_data[block_unpredictable_count ++] = curData;
									}		
								}
								else{
									type[index] = 0;
									//pred = curData;  // no use
									unpredictable_data[block_unpredictable_count ++] = curData;
								}
								index ++;	
								cur_data_pos ++;
							}
							cur_data_pos ++;
						}
						cur_data_pos += pred_buffer_block_size;
					}
					reg_params ++;
					total_unpred += block_unpredictable_count;
					unpredictable_data += block_unpredictable_count;
					*blockwise_unpred_count = block_unpredictable_count;
				}
				else{
					// use SZ
					// SZ predication
					size_t unpredictable_count = 0;
					float * cur_data_pos = pred_buffer + pred_buffer_block_size * pred_buffer_block_size + pred_buffer_block_size + 1;
					float curData;
					float pred3D;
					double itvNum, diff;
					size_t index = 0;
					for(size_t ii=0; ii<block_size; ii++){
						for(size_t jj=0; jj<block_size; jj++){
							for(size_t kk=0; kk<block_size; kk++){
								curData = *cur_data_pos;
								pred3D = cur_data_pos[-1] + cur_data_pos[-strip_dim1_offset] + cur_data_pos[-strip_dim0_offset] - cur_data_pos[-strip_dim1_offset - 1]
									- cur_data_pos[-strip_dim0_offset - 1] - cur_data_pos[-strip_dim0_offset - strip_dim1_offset] + cur_data_pos[-strip_dim0_offset - strip_dim1_offset - 1];
								diff = curData - pred3D;
								itvNum = fabs(diff)/realPrecision + 1;
								if (itvNum < intvCapacity_sz){
									if (diff < 0) itvNum = -itvNum;
									type[index] = (int) (itvNum/2) + intvRadius;
									*cur_data_pos = pred3D + 2 * (type[index] - intvRadius) * realPrecision;
									//ganrantee comporession error against the case of machine-epsilon
									if(fabs(curData - *cur_data_pos)>realPrecision){	
										type[index] = 0;
										*cur_data_pos = curData;	
										unpredictable_data[unpredictable_count ++] = curData;
									}					
								}
								else{
									type[index] = 0;
									*cur_data_pos = curData;
									unpredictable_data[unpredictable_count ++] = curData;
								}
								index ++;
								cur_data_pos ++;
							}
							cur_data_pos ++;
						}
						cur_data_pos += pred_buffer_block_size;
					}
					total_unpred += unpredictable_count;
					unpredictable_data += unpredictable_count;
					*blockwise_unpred_count = unpredictable_count;
				}// end SZ
				blockwise_unpred_count ++;
				type += block_size * block_size * block_size;
			} // end k
			indicator += num_z;
		}// end j
	}// end i
}
