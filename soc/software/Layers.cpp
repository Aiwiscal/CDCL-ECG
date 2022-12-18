//
// Created by wenhan on 2022/1/24.
//

#include "Layers.h"
#include <cmath>

void Layers::conv1d(const float *inputs, float *outputs, float *weight, const float *bias, int inputSize, int outputSize,
                    int kernelSize, int inputChannels, int outputChannels, int stride, int pad) {

    for (int i = 0; i < outputChannels; ++i) {
        float *pw = weight + i * inputChannels * kernelSize;
        float b = bias[i];
        float *po = outputs + i * outputSize;
        for (int j = 0; j < outputSize; ++j) {
            float tmp = b;
            for (int k = 0; k < inputChannels; ++k) {
                for (int l = 0; l < kernelSize; ++l) {
                    int h = j * stride + l - pad;
                    if ((h >= 0) && (h < inputSize)){
                        tmp += inputs[k * inputSize + h] * pw[k * kernelSize + l];
                    }
                }
            }
            po[j] = tmp;
        }
    }
}

void Layers::bn(const float *inputs, float *outputs, const float *weight, const float *bias,
                const float *mean, const float *var, int inputSize, int inputChannels) {
    for (int i = 0; i < inputChannels; ++i) {
        float w = weight[i];
        float b = bias[i];
        float m = mean[i];
        float v = var[i];
        for (int j = 0; j < inputSize; ++j) {
            float t = w * ((inputs[i*inputSize+j] - m) / sqrt(v + 0.001f)) + b;
            outputs[i*inputSize+j] = (t > 0.0f)?t:0.0f;
        }
    }
}

void Layers::fc(const float *inputs, float *outputs, const float *weight, const float *bias, int inputSize, int outputSize) {
    for (int i = 0; i < outputSize; ++i) {
        float b = bias[i];
        float tmp = b;
        for (int j = 0; j < inputSize; ++j) {
            float w = weight[i*inputSize + j];
            tmp += w * inputs[j];
        }
        outputs[i] = tmp;
    }
}
