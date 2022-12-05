#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "sample.h"
#include "encoder_net.h"

using namespace aocl_utils;

// OpenCL runtime configuration
cl_platform_id platform = NULL;
unsigned num_devices = 0;
scoped_array<cl_device_id> device; // num_devices elements
cl_context context = NULL;
cl_command_queue queue;
cl_program program = NULL;

cl_kernel conv1d, bn;
cl_mem d_sample, d_conv0_weight, d_conv0_bias, d_conv0_out;
cl_mem d_bn0_weight, d_bn0_bias, d_bn0_mean, d_bn0_var, d_bn0_out;
cl_mem d_conv1_weight, d_conv1_bias, d_conv1_out;
cl_mem d_bn1_weight, d_bn1_bias, d_bn1_mean, d_bn1_var, d_bn1_out;
cl_mem d_conv2_weight, d_conv2_bias, d_conv2_out;
cl_mem d_bn2_weight, d_bn2_bias, d_bn2_mean, d_bn2_var, d_bn2_out;
cl_mem d_conv3_weight, d_conv3_bias, d_conv3_out;
cl_mem d_bn3_weight, d_bn3_bias, d_bn3_mean, d_bn3_var, d_bn3_out;
cl_mem d_conv4_weight, d_conv4_bias, d_conv4_out;
cl_mem d_bn4_weight, d_bn4_bias, d_bn4_mean, d_bn4_var, d_bn4_out;
cl_mem d_conv5_weight, d_conv5_bias, d_conv5_out;
cl_mem d_bn5_weight, d_bn5_bias, d_bn5_mean, d_bn5_var, d_bn5_out;


cl_event conv0_event, bn0_event; 
cl_event conv1_event, bn1_event;
cl_event conv2_event, bn2_event;
cl_event conv3_event, bn3_event;
cl_event conv4_event, bn4_event;
cl_event conv5_event, bn5_event;

float *ret = (float *)malloc((output_size_5 * output_channels_5) * sizeof(float)); 
void write_outputs();
void cleanup();

int main()
{
	cl_int status;

	printf("Initializing OpenCL\n");

	if(!setCwdToExeDir()) {
	    return 1;
	}

	  // Get the OpenCL platform.
	platform = findPlatform("Intel(R) FPGA");
	if(platform == NULL) {
	  printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
	  return 1;
	}

	  // Query the available OpenCL device.
	  device.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
	  printf("Platform: %s\n", getPlatformName(platform).c_str());
	  printf("Using %d device(s)\n", num_devices);
	  for(unsigned int i = 0; i < num_devices; ++i) {
	    printf("  %s\n", getDeviceName(device[i]).c_str());
	  }

	  // Create the context.
	  context = clCreateContext(NULL, num_devices, device, NULL, NULL, &status);
	  checkError(status, "Failed to create context");

	std::string binary_file = getBoardBinaryFile("encoder_net", device[0]);
	printf("Using AOCX: %s\n", binary_file.c_str());
	program = createProgramFromBinary(context, binary_file.c_str(), device, num_devices);

	status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
	checkError(status, "Failed to build program");

	queue = clCreateCommandQueue(context, device[0], CL_QUEUE_PROFILING_ENABLE, &status);
	checkError(status, "Failed to create command queue");

	// conv1d_input = clCreateKernel(program, "conv1d_input" , &status);
	// checkError(status, "Failed to create kernel conv1d_input");

	conv1d = clCreateKernel(program, "conv1d", &status);
	checkError(status, "Failed to create kernel conv1d");

	bn = clCreateKernel(program, "batch_norm", &status);
	checkError(status, "Failed to create kernel batchnorm");

	d_sample = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input_size_0*input_channels_0*sizeof(float), sample, &status);
	checkError(status, "Failed to create d_sample buffer");

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------
	d_conv0_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input_channels_0*filter_length_0*output_channels_0*sizeof(float), conv0_weight, &status);
	checkError(status, "Failed to create d_conv0_weight buffer");

	d_conv0_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, output_channels_0*sizeof(float), conv0_bias, &status);
	checkError(status, "Failed to create d_conv0_bias buffer");

	d_conv0_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_size_0*output_channels_0*sizeof(float), NULL, &status);
	checkError(status, "Failed to create d_conv0_out buffer");

	d_bn0_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, output_channels_0*sizeof(float), bn0_gamma, &status);
	checkError(status, "Failed to create d_bn0_weight buffer");

	d_bn0_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, output_channels_0*sizeof(float), bn0_beta, &status);
	checkError(status, "Failed to create d_bn0_bias buffer");

	d_bn0_mean = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, output_channels_0*sizeof(float), bn0_mean, &status);
	checkError(status, "Failed to create d_bn0_mean buffer");

	d_bn0_var = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, output_channels_0*sizeof(float), bn0_var, &status);
	checkError(status, "Failed to create d_bn0_var buffer");

	d_bn0_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_size_0*output_channels_0*sizeof(float), NULL, &status);
	checkError(status, "Failed to create d_bn0_out buffer");

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------

	d_conv1_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input_channels_1*filter_length_1*output_channels_1*sizeof(float), conv1_weight, &status);
	checkError(status, "Failed to create d_conv1_weight buffer");

	d_conv1_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, output_channels_1*sizeof(float), conv1_bias, &status);
	checkError(status, "Failed to create d_conv1_bias buffer");

	d_conv1_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_size_1*output_channels_1*sizeof(float), NULL, &status);
	checkError(status, "Failed to create d_conv1_out buffer");

	d_bn1_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, output_channels_1*sizeof(float), bn1_gamma, &status);
	checkError(status, "Failed to create d_bn1_weight buffer");

	d_bn1_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, output_channels_1*sizeof(float), bn1_beta, &status);
	checkError(status, "Failed to create d_bn1_bias buffer");

	d_bn1_mean = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, output_channels_1*sizeof(float), bn1_mean, &status);
	checkError(status, "Failed to create d_bn1_mean buffer");

	d_bn1_var = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, output_channels_1*sizeof(float), bn1_var, &status);
	checkError(status, "Failed to create d_bn1_var buffer");

	d_bn1_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_size_1*output_channels_1*sizeof(float), NULL, &status);
	checkError(status, "Failed to create d_bn1_out buffer");

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------

	d_conv2_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input_channels_2*filter_length_2*output_channels_2*sizeof(float), conv2_weight, &status);
	checkError(status, "Failed to create d_conv2_weight buffer");

	d_conv2_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, output_channels_2*sizeof(float), conv2_bias, &status);
	checkError(status, "Failed to create d_conv2_bias buffer");

	d_conv2_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_size_2*output_channels_2*sizeof(float), NULL, &status);
	checkError(status, "Failed to create d_conv2_out buffer");

	d_bn2_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, output_channels_2*sizeof(float), bn2_gamma, &status);
	checkError(status, "Failed to create d_bn2_weight buffer");

	d_bn2_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, output_channels_2*sizeof(float), bn2_beta, &status);
	checkError(status, "Failed to create d_bn2_bias buffer");

	d_bn2_mean = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, output_channels_2*sizeof(float), bn2_mean, &status);
	checkError(status, "Failed to create d_bn2_mean buffer");

	d_bn2_var = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, output_channels_2*sizeof(float), bn2_var, &status);
	checkError(status, "Failed to create d_bn2_var buffer");

	d_bn2_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_size_2*output_channels_2*sizeof(float), NULL, &status);
	checkError(status, "Failed to create d_bn2_out buffer");

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------

	d_conv3_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input_channels_3*filter_length_3*output_channels_3*sizeof(float), conv3_weight, &status);
	checkError(status, "Failed to create d_conv3_weight buffer");

	d_conv3_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, output_channels_3*sizeof(float), conv3_bias, &status);
	checkError(status, "Failed to create d_conv3_bias buffer");

	d_conv3_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_size_3*output_channels_3*sizeof(float), NULL, &status);
	checkError(status, "Failed to create d_conv3_out buffer");

	d_bn3_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, output_channels_3*sizeof(float), bn3_gamma, &status);
	checkError(status, "Failed to create d_bn3_weight buffer");

	d_bn3_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, output_channels_3*sizeof(float), bn3_beta, &status);
	checkError(status, "Failed to create d_bn3_bias buffer");

	d_bn3_mean = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, output_channels_3*sizeof(float), bn3_mean, &status);
	checkError(status, "Failed to create d_bn3_mean buffer");

	d_bn3_var = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, output_channels_3*sizeof(float), bn3_var, &status);
	checkError(status, "Failed to create d_bn3_var buffer");

	d_bn3_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_size_3*output_channels_3*sizeof(float), NULL, &status);
	checkError(status, "Failed to create d_bn3_out buffer");

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------

	d_conv4_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input_channels_4*filter_length_4*output_channels_4*sizeof(float), conv4_weight, &status);
	checkError(status, "Failed to create d_conv4_weight buffer");

	d_conv4_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, output_channels_4*sizeof(float), conv4_bias, &status);
	checkError(status, "Failed to create d_conv4_bias buffer");

	d_conv4_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_size_4*output_channels_4*sizeof(float), NULL, &status);
	checkError(status, "Failed to create d_conv4_out buffer");

	d_bn4_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, output_channels_4*sizeof(float), bn4_gamma, &status);
	checkError(status, "Failed to create d_bn4_weight buffer");

	d_bn4_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, output_channels_4*sizeof(float), bn4_beta, &status);
	checkError(status, "Failed to create d_bn4_bias buffer");

	d_bn4_mean = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, output_channels_4*sizeof(float), bn4_mean, &status);
	checkError(status, "Failed to create d_bn4_mean buffer");

	d_bn4_var = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, output_channels_4*sizeof(float), bn4_var, &status);
	checkError(status, "Failed to create d_bn4_var buffer");

	d_bn4_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_size_4*output_channels_4*sizeof(float), NULL, &status);
	checkError(status, "Failed to create d_bn4_out buffer");

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------

	d_conv5_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input_channels_5*filter_length_5*output_channels_5*sizeof(float), conv5_weight, &status);
	checkError(status, "Failed to create d_conv5_weight buffer");

	d_conv5_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, output_channels_5*sizeof(float), conv5_bias, &status);
	checkError(status, "Failed to create d_conv5_bias buffer");

	d_conv5_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_size_5*output_channels_5*sizeof(float), NULL, &status);
	checkError(status, "Failed to create d_conv5_out buffer");

	d_bn5_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, output_channels_5*sizeof(float), bn5_gamma, &status);
	checkError(status, "Failed to create d_bn5_weight buffer");

	d_bn5_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, output_channels_5*sizeof(float), bn5_beta, &status);
	checkError(status, "Failed to create d_bn5_bias buffer");

	d_bn5_mean = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, output_channels_5*sizeof(float), bn5_mean, &status);
	checkError(status, "Failed to create d_bn5_mean buffer");

	d_bn5_var = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, output_channels_5*sizeof(float), bn5_var, &status);
	checkError(status, "Failed to create d_bn5_var buffer");

	d_bn5_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_size_5*output_channels_5*sizeof(float), NULL, &status);
	checkError(status, "Failed to create d_bn5_out buffer");

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------


	printf("\r\nEncoderNet on FPGA start:\r\n");
	for (int t = 0; t < 10; ++t)
	{
		/* code */
		double start_time = getCurrentTimestamp();

		status |= clSetKernelArg(conv1d, 0, sizeof(int), &(input_channels_0));
	    status |= clSetKernelArg(conv1d, 1, sizeof(int), &(input_size_0));
	    status |= clSetKernelArg(conv1d, 2, sizeof(int), &(pad_0));
	    status |= clSetKernelArg(conv1d, 3, sizeof(int), &(stride_0));
	    status |= clSetKernelArg(conv1d, 4, sizeof(int), &(output_size_0));
	    status |= clSetKernelArg(conv1d, 5, sizeof(int), &(filter_length_0));
	    status |= clSetKernelArg(conv1d, 6, sizeof(cl_mem), &d_sample);
	    status |= clSetKernelArg(conv1d, 7, sizeof(cl_mem), &d_conv0_weight);
	    status |= clSetKernelArg(conv1d, 8, sizeof(cl_mem), &d_conv0_bias);
	    status |= clSetKernelArg(conv1d, 9, sizeof(cl_mem), &d_conv0_out);
	    checkError(status, "Setting conv_input: conv_input arguments");
	    size_t conv0_work_size[] = {16, 1024};
	    status = clEnqueueNDRangeKernel(queue, conv1d, 2, NULL, conv0_work_size, NULL, 0, NULL, &conv0_event);
	    checkError(status, "Enqueueing conv_input 0");


		status |= clSetKernelArg(bn, 0, sizeof(int), &output_size_0);
		status |= clSetKernelArg(bn, 1, sizeof(cl_mem), &d_conv0_out);
		status |= clSetKernelArg(bn, 2, sizeof(cl_mem), &d_bn0_weight);
		status |= clSetKernelArg(bn, 3, sizeof(cl_mem), &d_bn0_bias);
		status |= clSetKernelArg(bn, 4, sizeof(cl_mem), &d_bn0_mean);
		status |= clSetKernelArg(bn, 5, sizeof(cl_mem), &d_bn0_var);
		status |= clSetKernelArg(bn, 6, sizeof(cl_mem), &d_bn0_out);
		checkError(status, "Setting batchnorm 0 arguments");
		size_t bn0_work_size[] = {16, 1024};
		status = clEnqueueNDRangeKernel(queue, bn, 2, NULL, bn0_work_size, NULL, 1, &conv0_event, &bn0_event);
	    checkError(status, "Enqueueing bn 0");

	    status |= clSetKernelArg(conv1d, 0, sizeof(int), &(input_channels_1));
	    status |= clSetKernelArg(conv1d, 1, sizeof(int), &(input_size_1));
	    status |= clSetKernelArg(conv1d, 2, sizeof(int), &(pad_1));
	    status |= clSetKernelArg(conv1d, 3, sizeof(int), &(stride_1));
	    status |= clSetKernelArg(conv1d, 4, sizeof(int), &(output_size_1));
	    status |= clSetKernelArg(conv1d, 5, sizeof(int), &(filter_length_1));
	    status |= clSetKernelArg(conv1d, 6, sizeof(cl_mem), &(d_bn0_out));
	    status |= clSetKernelArg(conv1d, 7, sizeof(cl_mem), &(d_conv1_weight));
	    status |= clSetKernelArg(conv1d, 8, sizeof(cl_mem), &(d_conv1_bias));
	    status |= clSetKernelArg(conv1d, 9, sizeof(cl_mem), &(d_conv1_out));
	    checkError(status, "Setting conv: conv arguments");
	    size_t conv1_work_size[] = {32, 512};
	    status = clEnqueueNDRangeKernel(queue, conv1d, 2, NULL, conv1_work_size, NULL, 1, &bn0_event, &conv1_event);
	    checkError(status, "Enqueueing conv 1");

	    status |= clSetKernelArg(bn, 0, sizeof(int), &output_size_1);
	    status |= clSetKernelArg(bn, 1, sizeof(cl_mem), &d_conv1_out);
	    status |= clSetKernelArg(bn, 2, sizeof(cl_mem), &d_bn1_weight);
	    status |= clSetKernelArg(bn, 3, sizeof(cl_mem), &d_bn1_bias);
	    status |= clSetKernelArg(bn, 4, sizeof(cl_mem), &d_bn1_mean);
		status |= clSetKernelArg(bn, 5, sizeof(cl_mem), &d_bn1_var);
		status |= clSetKernelArg(bn, 6, sizeof(cl_mem), &d_bn1_out);
		checkError(status, "Setting batchnorm 1 arguments");
		size_t bn1_work_size[] = {32, 512};
		status = clEnqueueNDRangeKernel(queue, bn, 2, NULL, bn1_work_size, NULL, 1, &conv1_event, &bn1_event);
		checkError(status, "Enqueueing bn 1");

		status |= clSetKernelArg(conv1d, 0, sizeof(int), &(input_channels_2));
	    status |= clSetKernelArg(conv1d, 1, sizeof(int), &(input_size_2));
	    status |= clSetKernelArg(conv1d, 2, sizeof(int), &(pad_2));
	    status |= clSetKernelArg(conv1d, 3, sizeof(int), &(stride_2));
	    status |= clSetKernelArg(conv1d, 4, sizeof(int), &(output_size_2));
	    status |= clSetKernelArg(conv1d, 5, sizeof(int), &(filter_length_2));
	    status |= clSetKernelArg(conv1d, 6, sizeof(cl_mem), &d_bn1_out);
	    status |= clSetKernelArg(conv1d, 7, sizeof(cl_mem), &d_conv2_weight);
	    status |= clSetKernelArg(conv1d, 8, sizeof(cl_mem), &d_conv2_bias);
	    status |= clSetKernelArg(conv1d, 9, sizeof(cl_mem), &d_conv2_out);
	    checkError(status, "Setting conv: conv arguments");
	    size_t conv2_work_size[] = {64, 256};
	    status = clEnqueueNDRangeKernel(queue, conv1d, 2, NULL, conv2_work_size, NULL, 1, &bn1_event, &conv2_event);
	    checkError(status, "Enqueueing conv 2");

	    status |= clSetKernelArg(bn, 0, sizeof(int), &output_size_2);
	    status |= clSetKernelArg(bn, 1, sizeof(cl_mem), &d_conv2_out);
	    status |= clSetKernelArg(bn, 2, sizeof(cl_mem), &d_bn2_weight);
	    status |= clSetKernelArg(bn, 3, sizeof(cl_mem), &d_bn2_bias);
	    status |= clSetKernelArg(bn, 4, sizeof(cl_mem), &d_bn2_mean);
		status |= clSetKernelArg(bn, 5, sizeof(cl_mem), &d_bn2_var);
		status |= clSetKernelArg(bn, 6, sizeof(cl_mem), &d_bn2_out);
		checkError(status, "Setting batchnorm 2 arguments");
		size_t bn2_work_size[] = {64, 256};
		status = clEnqueueNDRangeKernel(queue, bn, 2, NULL, bn2_work_size, NULL, 1, &conv2_event, &bn2_event);
		checkError(status, "Enqueueing bn 2");

		status |= clSetKernelArg(conv1d, 0, sizeof(int), &(input_channels_3));
	    status |= clSetKernelArg(conv1d, 1, sizeof(int), &(input_size_3));
	    status |= clSetKernelArg(conv1d, 2, sizeof(int), &(pad_3));
	    status |= clSetKernelArg(conv1d, 3, sizeof(int), &(stride_3));
	    status |= clSetKernelArg(conv1d, 4, sizeof(int), &(output_size_3));
	    status |= clSetKernelArg(conv1d, 5, sizeof(int), &(filter_length_3));
	    status |= clSetKernelArg(conv1d, 6, sizeof(cl_mem), &d_bn2_out);
	    status |= clSetKernelArg(conv1d, 7, sizeof(cl_mem), &d_conv3_weight);
	    status |= clSetKernelArg(conv1d, 8, sizeof(cl_mem), &d_conv3_bias);
	    status |= clSetKernelArg(conv1d, 9, sizeof(cl_mem), &d_conv3_out);
	    checkError(status, "Setting conv: conv arguments");
	    size_t conv3_work_size[] = {64, 128};
	    status = clEnqueueNDRangeKernel(queue, conv1d, 2, NULL, conv3_work_size, NULL, 1, &bn2_event, &conv3_event);
	    checkError(status, "Enqueueing conv 3");

	    status |= clSetKernelArg(bn, 0, sizeof(int), &output_size_3);
	    status |= clSetKernelArg(bn, 1, sizeof(cl_mem), &d_conv3_out);
	    status |= clSetKernelArg(bn, 2, sizeof(cl_mem), &d_bn3_weight);
	    status |= clSetKernelArg(bn, 3, sizeof(cl_mem), &d_bn3_bias);
	    status |= clSetKernelArg(bn, 4, sizeof(cl_mem), &d_bn3_mean);
		status |= clSetKernelArg(bn, 5, sizeof(cl_mem), &d_bn3_var);
		status |= clSetKernelArg(bn, 6, sizeof(cl_mem), &d_bn3_out);
		checkError(status, "Setting batchnorm 3 arguments");
		size_t bn3_work_size[] = {64, 128};
		status = clEnqueueNDRangeKernel(queue, bn, 2, NULL, bn3_work_size, NULL, 1, &conv3_event, &bn3_event);
		checkError(status, "Enqueueing bn 3");


		status |= clSetKernelArg(conv1d, 0, sizeof(int), &(input_channels_4));
	    status |= clSetKernelArg(conv1d, 1, sizeof(int), &(input_size_4));
	    status |= clSetKernelArg(conv1d, 2, sizeof(int), &(pad_4));
	    status |= clSetKernelArg(conv1d, 3, sizeof(int), &(stride_4));
	    status |= clSetKernelArg(conv1d, 4, sizeof(int), &(output_size_4));
	    status |= clSetKernelArg(conv1d, 5, sizeof(int), &(filter_length_4));
	    status |= clSetKernelArg(conv1d, 6, sizeof(cl_mem), &d_bn3_out);
	    status |= clSetKernelArg(conv1d, 7, sizeof(cl_mem), &d_conv4_weight);
	    status |= clSetKernelArg(conv1d, 8, sizeof(cl_mem), &d_conv4_bias);
	    status |= clSetKernelArg(conv1d, 9, sizeof(cl_mem), &d_conv4_out);
	    checkError(status, "Setting conv: conv arguments");
	    size_t conv4_work_size[] = {128, 128};
	    status = clEnqueueNDRangeKernel(queue, conv1d, 2, NULL, conv4_work_size, NULL, 1, &bn3_event, &conv4_event);
	    checkError(status, "Enqueueing conv 4");

	    status |= clSetKernelArg(bn, 0, sizeof(int), &output_size_4);
	    status |= clSetKernelArg(bn, 1, sizeof(cl_mem), &d_conv4_out);
	    status |= clSetKernelArg(bn, 2, sizeof(cl_mem), &d_bn4_weight);
	    status |= clSetKernelArg(bn, 3, sizeof(cl_mem), &d_bn4_bias);
	    status |= clSetKernelArg(bn, 4, sizeof(cl_mem), &d_bn4_mean);
		status |= clSetKernelArg(bn, 5, sizeof(cl_mem), &d_bn4_var);
		status |= clSetKernelArg(bn, 6, sizeof(cl_mem), &d_bn4_out);
		checkError(status, "Setting batchnorm 4 arguments");
		size_t bn4_work_size[] = {128, 128};
		status = clEnqueueNDRangeKernel(queue, bn, 2, NULL, bn4_work_size, NULL, 1, &conv4_event, &bn4_event);
		checkError(status, "Enqueueing bn 4");

		status |= clSetKernelArg(conv1d, 0, sizeof(int), &(input_channels_5));
	    status |= clSetKernelArg(conv1d, 1, sizeof(int), &(input_size_5));
	    status |= clSetKernelArg(conv1d, 2, sizeof(int), &(pad_5));
	    status |= clSetKernelArg(conv1d, 3, sizeof(int), &(stride_5));
	    status |= clSetKernelArg(conv1d, 4, sizeof(int), &(output_size_5));
	    status |= clSetKernelArg(conv1d, 5, sizeof(int), &(filter_length_5));
	    status |= clSetKernelArg(conv1d, 6, sizeof(cl_mem), &d_bn4_out);
	    status |= clSetKernelArg(conv1d, 7, sizeof(cl_mem), &d_conv5_weight);
	    status |= clSetKernelArg(conv1d, 8, sizeof(cl_mem), &d_conv5_bias);
	    status |= clSetKernelArg(conv1d, 9, sizeof(cl_mem), &d_conv5_out);
	    checkError(status, "Setting conv: conv arguments");
	    size_t conv5_work_size[] = {256, 128};
	    status = clEnqueueNDRangeKernel(queue, conv1d, 2, NULL, conv5_work_size, NULL, 1, &bn4_event, &conv5_event);
	    checkError(status, "Enqueueing conv 5");

	    status |= clSetKernelArg(bn, 0, sizeof(int), &output_size_5);
	    status |= clSetKernelArg(bn, 1, sizeof(cl_mem), &d_conv5_out);
	    status |= clSetKernelArg(bn, 2, sizeof(cl_mem), &d_bn5_weight);
	    status |= clSetKernelArg(bn, 3, sizeof(cl_mem), &d_bn5_bias);
	    status |= clSetKernelArg(bn, 4, sizeof(cl_mem), &d_bn5_mean);
		status |= clSetKernelArg(bn, 5, sizeof(cl_mem), &d_bn5_var);
		status |= clSetKernelArg(bn, 6, sizeof(cl_mem), &d_bn5_out);
		checkError(status, "Setting batchnorm 5 arguments");
		size_t bn5_work_size[] = {256, 128};
		status = clEnqueueNDRangeKernel(queue, bn, 2, NULL, bn5_work_size, NULL, 1, &conv5_event, &bn5_event);
		checkError(status, "Enqueueing bn 5");



	    // status = clFinish(queue);
	    // checkError(status, "Wait for finish");
	    clWaitForEvents(1,&bn5_event);

	    status = clEnqueueReadBuffer(queue, d_bn5_out, CL_TRUE, 0, output_size_5*output_channels_5*sizeof(float), ret, 0, NULL, NULL );
	    double end_time = getCurrentTimestamp();
	    printf("\r\nrunning time: %0.3f ms\r\n", (end_time - start_time) * 1e3);

	    cl_ulong conv0_time = getStartEndTime(conv0_event);
	    cl_ulong bn0_time = getStartEndTime(bn0_event);
	    cl_ulong conv1_time = getStartEndTime(conv1_event);
	    cl_ulong bn1_time = getStartEndTime(bn1_event);
	    cl_ulong conv2_time = getStartEndTime(conv2_event);
	    cl_ulong bn2_time = getStartEndTime(bn2_event);
	    cl_ulong conv3_time = getStartEndTime(conv3_event);
	    cl_ulong bn3_time = getStartEndTime(bn3_event);
	    cl_ulong conv4_time = getStartEndTime(conv4_event);
	    cl_ulong bn4_time = getStartEndTime(bn4_event);
	    cl_ulong conv5_time = getStartEndTime(conv5_event);
	    cl_ulong bn5_time = getStartEndTime(bn5_event);

	    cl_ulong total_time = conv0_time + bn0_time + conv1_time + bn1_time + conv2_time + bn2_time + conv3_time + bn3_time + conv4_time + bn4_time + conv5_time + bn5_time;

	    printf("conv0_time: %0.3f ms, bn0_time: %0.3f ms\n", conv0_time*1e-6, bn0_time*1e-6);
	    printf("conv1_time: %0.3f ms, bn1_time: %0.3f ms\n", conv1_time*1e-6, bn1_time*1e-6);
	    printf("conv2_time: %0.3f ms, bn2_time: %0.3f ms\n", conv2_time*1e-6, bn2_time*1e-6);
	    printf("conv3_time: %0.3f ms, bn3_time: %0.3f ms\n", conv3_time*1e-6, bn3_time*1e-6);
	    printf("conv4_time: %0.3f ms, bn4_time: %0.3f ms\n", conv4_time*1e-6, bn4_time*1e-6);
	    printf("conv5_time: %0.3f ms, bn5_time: %0.3f ms\n", conv5_time*1e-6, bn5_time*1e-6);

	    printf("total_time: %0.3f ms\n", total_time * 1e-6);
	    checkError(status, "Failed to read output");
	}
    
    write_outputs();
    cleanup();

	return 0;
}

void write_outputs(){
	FILE* fp = fopen("outputs.txt", "w");
	for(int i=0; i<output_size_5 * output_channels_5;i++){
		fprintf(fp, "%f\n", ret[i]);
	}
	fclose(fp);
}

void cleanup(){
	clReleaseEvent(conv0_event);
	clReleaseEvent(bn0_event);
	clReleaseEvent(conv1_event);
	clReleaseEvent(bn1_event);
	clReleaseEvent(conv2_event);
	clReleaseEvent(bn2_event);
	clReleaseEvent(conv3_event);
	clReleaseEvent(bn3_event);
	clReleaseEvent(conv4_event);
	clReleaseEvent(bn4_event);
	clReleaseEvent(conv5_event);
	clReleaseEvent(bn5_event);
	clReleaseMemObject(d_sample);
  	clReleaseMemObject(d_conv0_weight);
  	clReleaseMemObject(d_conv0_bias);
  	clReleaseMemObject(d_conv0_out);
	clReleaseMemObject(d_bn0_weight);
	clReleaseMemObject(d_bn0_bias);
	clReleaseMemObject(d_bn0_mean);
	clReleaseMemObject(d_bn0_var);
	clReleaseMemObject(d_bn0_out);
	clReleaseMemObject(d_conv1_weight);
  	clReleaseMemObject(d_conv1_bias);
  	clReleaseMemObject(d_conv1_out);
	clReleaseMemObject(d_bn1_weight);
	clReleaseMemObject(d_bn1_bias);
	clReleaseMemObject(d_bn1_mean);
	clReleaseMemObject(d_bn1_var);
	clReleaseMemObject(d_bn1_out);
	clReleaseMemObject(d_conv2_weight);
  	clReleaseMemObject(d_conv2_bias);
  	clReleaseMemObject(d_conv2_out);
	clReleaseMemObject(d_bn2_weight);
	clReleaseMemObject(d_bn2_bias);
	clReleaseMemObject(d_bn2_mean);
	clReleaseMemObject(d_bn2_var);
	clReleaseMemObject(d_bn2_out);
	clReleaseMemObject(d_conv3_weight);
  	clReleaseMemObject(d_conv3_bias);
  	clReleaseMemObject(d_conv3_out);
	clReleaseMemObject(d_bn3_weight);
	clReleaseMemObject(d_bn3_bias);
	clReleaseMemObject(d_bn3_mean);
	clReleaseMemObject(d_bn3_var);
	clReleaseMemObject(d_bn3_out);
	clReleaseMemObject(d_conv4_weight);
  	clReleaseMemObject(d_conv4_bias);
  	clReleaseMemObject(d_conv4_out);
	clReleaseMemObject(d_bn4_weight);
	clReleaseMemObject(d_bn4_bias);
	clReleaseMemObject(d_bn4_mean);
	clReleaseMemObject(d_bn4_var);
	clReleaseMemObject(d_bn4_out);
	clReleaseMemObject(d_conv5_weight);
  	clReleaseMemObject(d_conv5_bias);
  	clReleaseMemObject(d_conv5_out);
	clReleaseMemObject(d_bn5_weight);
	clReleaseMemObject(d_bn5_bias);
	clReleaseMemObject(d_bn5_mean);
	clReleaseMemObject(d_bn5_var);
	clReleaseMemObject(d_bn5_out);

	clReleaseKernel(conv1d);
	clReleaseKernel(bn);

	clReleaseCommandQueue(queue);
	clReleaseProgram(program);
	clReleaseContext(context);

	free(ret);

}