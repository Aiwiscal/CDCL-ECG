# -*- coding: utf-8 -*-
# @Author   : WenHan

# cnn_net_2D.py
import keras

# 2-dimensional backbone


def build_net(input_shape, n_feature_maps, nb_classes):
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv2D(filters=n_feature_maps, kernel_size=(11, 11), strides=(3, 3), padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv2D(filters=n_feature_maps * 2, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv2D(filters=n_feature_maps * 4, kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv2D(filters=n_feature_maps * 4, kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv2D(filters=n_feature_maps * 8, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv2D(filters=n_feature_maps * 16, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    feature = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(nb_classes, activation='softmax')(feature)

    return inputs, outputs, feature


if __name__ == '__main__':
    # set up a 2-d backbone:
    my_inputs, my_outputs, my_feature = build_net([300, 300, 1], 16, 5)
    backbone = keras.models.Model(inputs=my_inputs, outputs=my_feature)
    # check the architecture:
    backbone.summary()
