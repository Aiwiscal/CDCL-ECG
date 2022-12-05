#define NUM_LEADS 12
#define INPUT_FILTER_LENGTH 11
#define EPS 0.001

__kernel void conv1d(
    const int input_channels,
    const int input_size,
    const int pad,
    const int stride,
    const int output_size,
    const int filter_length,
    __global float *restrict input_im,
    __global const float *restrict filter_weight,
    __global const float *restrict filter_bias,
    __global float *restrict output_im
    )
{
    int filter_index = get_global_id(0);
    int i =  get_global_id(1);

    filter_weight += filter_index * input_channels * filter_length;
    float bias = filter_bias[filter_index];
    output_im +=filter_index * output_size;

    
	float tmp = bias;
	int h_start = i * stride - pad;
	
	for(int k = 0; k < input_channels; k++)
	{
		int k_start = k * input_size;
		int w_start = k * filter_length;
		#pragma unroll 11
		for(int l = 0; l < filter_length; l++)
		{
			int h = h_start + l;
			if((h >= 0) && (h < input_size))
			{
				tmp += input_im[k_start + h] * filter_weight[w_start + l];
			}
		}
	}

	output_im[i] = tmp;
    
}

__kernel void batch_norm(
                         const int in_size,
                         __global float *restrict in_data,
                        __global float *restrict bn_weights,
                        __global float *restrict bn_biases,
                        __global float *restrict bn_running_mean,
                        __global float *restrict bn_running_var,
                        __global float *restrict out_data
                        )
{
    int filter_index = get_global_id(0);
    int sample_index = get_global_id(1);
    int index = filter_index*in_size + sample_index;
    float out;

    // output = (alpha * ((input - mean)/sqrt(variance + eps)) + beta)
    out = (bn_weights[filter_index] * ((in_data[index] - bn_running_mean[filter_index])/sqrt(bn_running_var[filter_index] + EPS))) + bn_biases[filter_index] ;

    out_data[index] = (out > 0.0f) ? out : 0.0f; // ReLU
//    out_data[index] = out;
}
