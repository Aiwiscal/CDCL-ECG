# -*- coding: utf-8 -*-
# @Author   : WenHan

# cnn_net.py
import keras


# 1-dimensional backbone

def build_net(input_shape, n_feature_maps, nb_classes):
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=11, strides=2, padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=7, strides=2, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv1D(filters=n_feature_maps * 4, kernel_size=5, strides=2, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv1D(filters=n_feature_maps * 4, kernel_size=5, strides=2, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv1D(filters=n_feature_maps * 8, kernel_size=3, strides=1, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv1D(filters=n_feature_maps * 16, kernel_size=3, strides=1, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    feature = keras.layers.GlobalAveragePooling1D()(x)
    outputs = keras.layers.Dense(nb_classes, activation='softmax')(feature)

    return inputs, outputs, feature


if __name__ == '__main__':
    # set up a 1-d backbone:
    my_inputs, my_outputs, my_feature = build_net([2048, 12], 16, 5)
    backbone = keras.models.Model(inputs=my_inputs, outputs=my_feature)
    # check the architecture:
    backbone.summary()
