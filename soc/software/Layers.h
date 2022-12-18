//
// Created by wenhan on 2022/1/24.
//

#ifndef CONVNET_LAYERS_H
#define CONVNET_LAYERS_H

class Layers {
public:
    static void conv1d(const float *inputs, float *outputs, float *weight, const float *bias, int inputSize, int outputSize,
                       int kernelSize, int inputChannels, int outputChannels, int stride, int pad);

    static void bn(const float *inputs, float *outputs, const float *weight, const float *bias, const float *mean, const float *var,
                   int inputSize, int inputChannels);

    static void fc(const float *inputs, float *outputs, const float *weight, const float *bias, int inputSize, int outputSize);

};


#endif //CONVNET_LAYERS_H
