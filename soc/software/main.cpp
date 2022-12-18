#include <iostream>
#include <stdio.h>
#include <ctime>
#include "Layers.h"
#include "encoder_net.h"
#include "linear_classifier.h"
#include "sample.h"

float *convOut0 = new float [output_size_0 * output_channels_0];
float *bnOut0 = new float [output_size_0 * output_channels_0];
float *convOut1 = new float [output_size_1 * output_channels_1];
float *bnOut1 = new float [output_size_1 * output_channels_1];
float *convOut2 = new float [output_size_2 * output_channels_2];
float *bnOut2 = new float [output_size_2 * output_channels_2];
float *convOut3 = new float [output_size_3 * output_channels_3];
float *bnOut3 = new float [output_size_3 * output_channels_3];
float *convOut4 = new float [output_size_4 * output_channels_4];
float *bnOut4 = new float [output_size_4 * output_channels_4];
float *convOut5 = new float [output_size_5 * output_channels_5];
float *bnOut5 = new float [output_size_5 * output_channels_5];
float *fcOut = new float [fc_output_size];
float *fcOut1 = new float [fc_output_size_1];

void cleanup();
void getResults();
int main() {
    // Only for inference time test !!!
    clock_t startTime, endTime;
    clock_t internalTime0, internalTime1, internalTime2, internalTime3, internalTime4;
    for (int i = 0; i < 2; ++i) {
        startTime = clock();
        Layers::conv1d(sample, convOut0, conv0_weight, conv0_bias, input_size_0, output_size_0, filter_length_0,
                       input_channels_0, output_channels_0, stride_0, pad_0);
        Layers::bn(convOut0, bnOut0, bn0_gamma, bn0_beta, bn0_mean, bn0_var, output_size_0, output_channels_0);
        internalTime0 = clock();

        Layers::conv1d(bnOut0, convOut1, conv1_weight, conv1_bias, input_size_1, output_size_1, filter_length_1,
                       input_channels_1, output_channels_1, stride_1, pad_1);
        Layers::bn(convOut1, bnOut1, bn1_gamma, bn1_beta, bn1_mean, bn1_var, output_size_1, output_channels_1);
        internalTime1 = clock();

        Layers::conv1d(bnOut1, convOut2, conv2_weight, conv2_bias, input_size_2, output_size_2, filter_length_2,
                       input_channels_2, output_channels_2, stride_2, pad_2);
        Layers::bn(convOut2, bnOut2, bn2_gamma, bn2_beta, bn2_mean, bn2_var, output_size_2, output_channels_2);
        internalTime2 = clock();

        Layers::conv1d(bnOut2, convOut3, conv3_weight, conv3_bias, input_size_3, output_size_3, filter_length_3,
                       input_channels_3, output_channels_3, stride_3, pad_3);
        Layers::bn(convOut3, bnOut3, bn3_gamma, bn3_beta, bn3_mean, bn3_var, output_size_3, output_channels_3);
        internalTime3 = clock();

        Layers::conv1d(bnOut3, convOut4, conv4_weight, conv4_bias, input_size_4, output_size_4, filter_length_4,
                       input_channels_4, output_channels_4, stride_4, pad_4);
        Layers::bn(convOut4, bnOut4, bn4_gamma, bn4_beta, bn4_mean, bn4_var, output_size_4, output_channels_4);
        internalTime4 = clock();

        Layers::conv1d(bnOut4, convOut5, conv5_weight, conv5_bias, input_size_5, output_size_5, filter_length_5,
                       input_channels_5, output_channels_5, stride_5, pad_5);
        Layers::bn(convOut5, bnOut5, bn5_gamma, bn5_beta, bn5_mean, bn5_var, output_size_5, output_channels_5);
        endTime = clock();
    }
    getResults();
    std::cout << "Layer 0 Time: " << (((double)( internalTime0 - startTime)) / CLOCKS_PER_SEC) * 1000 << " ms" << std::endl;
    std::cout << "Layer 1 Time: " << (((double)( internalTime1 - startTime)) / CLOCKS_PER_SEC) * 1000 << " ms" << std::endl;
    std::cout << "Layer 2 Time: " << (((double)( internalTime2 - startTime)) / CLOCKS_PER_SEC) * 1000 << " ms" << std::endl;
    std::cout << "Layer 3 Time: " << (((double)( internalTime3 - startTime)) / CLOCKS_PER_SEC) * 1000 << " ms" << std::endl;
    std::cout << "Layer 4 Time: " << (((double)( internalTime4 - startTime)) / CLOCKS_PER_SEC) * 1000 << " ms" << std::endl;
    std::cout << "Encoder network inference complete. Time:" << (((double)(endTime - startTime)) / CLOCKS_PER_SEC) * 1000 << " ms" << std::endl;

    // Only for inference time test !!!
    for (int i = 0; i < 1; ++i) {
        startTime = clock();
        Layers::fc(feat, fcOut1, fc_weights_1, fc_bias_1, fc_input_size, fc_output_size_1);
        endTime = clock();
    }
    for (int i = 0; i < fc_output_size; ++i) {
        std::cout << fcOut1[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Linear classifier inference complete. Time:" << (((double)(endTime - startTime)) / CLOCKS_PER_SEC) * 1000 << " ms" << std::endl;
    cleanup();
    return 0;
}

void cleanup(){
    delete [] convOut0;
    delete [] bnOut0;
    delete [] convOut1;
    delete [] bnOut1;
    delete [] convOut2;
    delete [] bnOut2;
    delete [] convOut3;
    delete [] bnOut3;
    delete [] convOut4;
    delete [] bnOut4;
    delete [] convOut5;
    delete [] bnOut5;
    delete [] fcOut;
    delete [] fcOut1;
}

void getResults(){
    FILE *fp = fopen("outputs.txt", "w");
    for (int i = 0; i < output_size_5 * output_channels_5; ++i) {
        fprintf(fp, "%f\n", bnOut5[i]);
    }
    fclose(fp);
}