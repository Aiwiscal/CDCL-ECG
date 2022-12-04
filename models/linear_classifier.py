# -*- coding: utf-8 -*-
# @Author   : WenHan

# linear_classifier.py
import tensorflow.keras as keras


# linear classifier
def build_net(input_shape, nb_classes):
    inputs = keras.layers.Input(shape=input_shape)
    outputs = keras.layers.Dense(units=nb_classes, activation="softmax")(inputs)
    return inputs, outputs


if __name__ == '__main__':
    num_classes = 5  # for PTB-XL
    # num_classes = 9  # for CPSC2018
    # Set up a linear classifier
    my_inputs, my_outputs = build_net([256, ], nb_classes=num_classes)
    my_model = keras.models.Model(inputs=my_inputs, outputs=my_outputs)
    # Check the linear classifier
    my_model.summary()
