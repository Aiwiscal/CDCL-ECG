# -*- coding: utf-8 -*-
# @Author   : WenHan
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model


def clr_model(seg_len_1d=2048, seg_len_2d=300):
    model = get_model(projection_layer_shapes=[256, 256],
                      ecg_len_1d=seg_len_1d, ecg_len_2d=seg_len_2d, num_channels=12)

    optimizer = keras.optimizers.Adam(1e-3)
    model.compile(
        loss=joint_contr_loss,
        optimizer=optimizer,
    )
    return model


def get_model(projection_layer_shapes, ecg_len_1d=2048, ecg_len_2d=300,
              num_channels=12):
    from models.cnn_net import build_net
    inputs_1d, _, x_1d = build_net([ecg_len_1d, num_channels], 16, 5)

    for i, projection_size in enumerate(projection_layer_shapes[:-1]):
        x_1d = Dense(
            projection_size,
            activation="relu",
            kernel_initializer="he_normal",
        )(x_1d)
    diagn_1d = Dense(
        projection_layer_shapes[-1],
        kernel_initializer="he_normal"
    )(x_1d)

    from models.cnn_net_2D import build_net
    inputs_2d, _, x_2d = build_net([ecg_len_2d, ecg_len_2d, 1], 16, 5)

    for i, projection_size in enumerate(projection_layer_shapes[:-1]):
        x_2d = Dense(
            projection_size,
            activation="relu",
            kernel_initializer="he_normal",
        )(x_2d)
    diagn_2d = Dense(
        projection_layer_shapes[-1],
        kernel_initializer="he_normal",
    )(x_2d)

    diagn = keras.layers.Concatenate(axis=0)([diagn_1d, diagn_2d])
    model = Model(
        [inputs_1d, inputs_2d],
        diagn
    )
    return model


def contr_loss(hidden1, hidden2):
    large_num = 1e9
    temperature = 0.1
    batch_size = tf.shape(hidden1)[0]
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    masks = tf.one_hot(
        tf.range(batch_size),
        batch_size,
    )

    logits_aa = tf.matmul(hidden1, hidden1, transpose_b=True) / temperature
    logits_aa = logits_aa - masks * large_num

    logits_bb = tf.matmul(hidden2, hidden2, transpose_b=True) / temperature
    logits_bb = logits_bb - masks * large_num

    logits_ab = tf.matmul(hidden1, hidden2, transpose_b=True) / temperature
    logits_ba = tf.matmul(hidden2, hidden1, transpose_b=True) / temperature
    loss_a = tf.nn.softmax_cross_entropy_with_logits(
        labels,
        tf.concat([logits_ab, logits_aa], 1),
    )
    loss_b = tf.nn.softmax_cross_entropy_with_logits(
        labels,
        tf.concat([logits_ba, logits_bb], 1),
    )
    return tf.add(loss_a, loss_b)


def joint_contr_loss(
        _,
        y_pred
):
    y_pred = tf.math.l2_normalize(y_pred, -1)
    hidden1_1d, hidden2_1d, hidden1_2d, hidden2_2d = tf.split(
        y_pred,
        4,
        0,
    )
    alpha = 0.5
    gamma = 0.5

    loss_1d = contr_loss(hidden1_1d, hidden2_1d) * alpha
    loss_cross = contr_loss(hidden1_1d, hidden1_2d) * gamma
    return loss_1d + loss_cross


if __name__ == '__main__':
    import os
    from data_utils.DataGenerator import CLRGenerator

    segment_length_1d = 2048
    segment_length_2d = 300
    epochs = 100
    model_out_path = "./encoders/"
    if not os.path.exists(model_out_path):
        os.mkdir(model_out_path)
    keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

    clr = clr_model(seg_len_1d=segment_length_1d, seg_len_2d=segment_length_2d)
    clr.summary()
    train_generator = CLRGenerator(data_path_1d="./data/ningbo_dataset/",
                                   data_path_2d="./data/ningbo_dataset/",
                                   use_data="train", batch_size=64)

    clr.fit_generator(train_generator, steps_per_epoch=len(train_generator), epochs=epochs)
    clr.save(model_out_path + "pre_train_model.h5")
