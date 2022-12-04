# -*- coding: utf-8 -*-
# @Author   : WenHan
import tensorflow.keras as keras
import cnn_net as cnn_net_1d
import cnn_net_2D as cnn_net_2d

if __name__ == '__main__':
    num_classes = 5  # for PTB-XL
    # num_classes = 9  # for CPSC2018
    # Set up a 1-d fully-supervised model
    # function build_net() is the same as the build_net() used in the first part of the response.
    fully_sup_1d_inputs, fully_sup_1d_outputs, _ = cnn_net_1d.build_net([2048, 12], 16, num_classes)
    fully_sup_1d_model = keras.models.Model(inputs=fully_sup_1d_inputs, outputs=fully_sup_1d_outputs)
    # Check the 1-d fully-supervised model
    fully_sup_1d_model.summary()

    # Set up a 2-d fully-supervised model
    # function build_net() is the same as the build_net() used in the first part of the response.
    fully_sup_2d_inputs, fully_sup_2d_outputs, _ = cnn_net_2d.build_net([300, 300, 1], 16, num_classes)
    fully_sup_2d_model = keras.models.Model(inputs=fully_sup_2d_inputs, outputs=fully_sup_2d_outputs)
    # Check the 2-d fully-supervised model
    fully_sup_2d_model.summary()
