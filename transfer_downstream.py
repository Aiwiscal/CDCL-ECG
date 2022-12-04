import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

nb_classes = 5  # for ptb-xl database.


# nb_classes = 9 # for cpsc2018 database.


class LinearProbe(object):
    def __init__(self, data_path, model_path, do_inference=True):
        self.data_path = data_path
        self.model_path = model_path
        self.do_inference = do_inference
        from pre_train import contr_loss, joint_contr_loss
        model = keras.models.load_model(model_path, custom_objects={"contr_loss": contr_loss,
                                                                    "joint_contr_loss": joint_contr_loss,
                                                                    "K": keras.backend})

        self.encoder_1d = keras.models.Model(inputs=model.inputs[0],
                                             outputs=model.get_layer("global_average_pooling1d").output)

    def infer_1d(self):
        if not self.do_inference:
            return
        print("start inference ...")
        print("load and preprocess train data ...")
        train_x = np.load(self.data_path + "X_train.npy")
        train_y = np.load(self.data_path + "y_train.npy")
        for i in tqdm(range(train_x.shape[0])):
            train_x[i] = (train_x[i] - np.mean(train_x[i])) / (np.std(train_x[i]) + 1e-6)
        print("infer train data ...")
        train_feat = self.encoder_1d.predict(train_x, verbose=1)
        print("train feat size: ", train_feat.shape)
        np.save("./train_feat.npy", train_feat)
        np.save("./train_y.npy", train_y)
        del train_x, train_y, train_feat

        print("load and preprocess val data ...")
        val_x = np.load(self.data_path + "X_val.npy")
        val_y = np.load(self.data_path + "y_val.npy")
        for i in tqdm(range(val_x.shape[0])):
            val_x[i] = (val_x[i] - np.mean(val_x[i])) / (np.std(val_x[i]) + 1e-6)
        print("infer val data ...")
        val_feat = self.encoder_1d.predict(val_x, verbose=1)
        print("val feat size: ", val_feat.shape)
        np.save("./val_feat.npy", val_feat)
        np.save("./val_y.npy", val_y)
        del val_x, val_y, val_feat

        print("load and preprocess test data ...")
        test_x = np.load(self.data_path + "X_test.npy")
        test_y = np.load(self.data_path + "y_test.npy")
        for i in tqdm(range(test_x.shape[0])):
            test_x[i] = (test_x[i] - np.mean(test_x[i])) / (np.std(test_x[i]) + 1e-6)
        print("infer test data ...")
        test_feat = self.encoder_1d.predict(test_x, verbose=1)
        print("test feat size: ", test_feat.shape)
        np.save("./test_feat.npy", test_feat)
        np.save("./test_y.npy", test_y)
        del test_x, test_y, test_feat

    def fit_eval(self, train_ratio=1.0):
        self.infer_1d()
        train_feat = np.load("./train_feat.npy")
        train_y = np.load("./train_y.npy")
        val_feat = np.load("./val_feat.npy")
        val_y = np.load("./val_y.npy")

        train_num = train_feat.shape[0]

        train_idx = np.arange(train_num)
        np.random.shuffle(train_idx)
        train_idx_select = train_idx[:int(train_num * train_ratio)]
        train_feat_select = train_feat[train_idx_select, ...]
        train_y_select = train_y[train_idx_select, ...]
        del train_feat, train_y
        print("train ratio = %f, train_select_shape = " % train_ratio, train_feat_select.shape, train_y_select.shape)

        keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()

        # linear classifier
        from models.linear_classifier import build_net
        inputs, outputs = build_net([train_feat_select.shape[-1], ], nb_classes)

        model = keras.models.Model(inputs=inputs, outputs=outputs)
        opt = keras.optimizers.Adam(lr=0.001)
        model_checkpoint = keras.callbacks.ModelCheckpoint("./linear_classifier_temp.h5", monitor="val_loss",
                                                           mode="auto", save_best_only=True)
        train_y_select = keras.utils.to_categorical(train_y_select, num_classes=nb_classes)
        val_y = keras.utils.to_categorical(val_y, num_classes=nb_classes)

        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        model.fit(x=train_feat_select, y=train_y_select, batch_size=128, epochs=100, verbose=2,
                  callbacks=[model_checkpoint],
                  validation_data=(val_feat, val_y))

        model = keras.models.load_model("./linear_classifier_temp.h5")
        test_feat = np.load("./test_feat.npy")
        test_y = np.load("./test_y.npy")

        test_y = keras.utils.to_categorical(test_y, num_classes=nb_classes)
        test_y_ = model.predict(test_feat, verbose=1)
        y = np.argmax(test_y, axis=1)
        y_ = np.argmax(test_y_, axis=1)

        acc = accuracy_score(y, y_)
        f1 = f1_score(y, y_, average="macro")
        conf_mat = confusion_matrix(y, y_)
        auc = roc_auc_score(test_y, test_y_, average="macro")
        print("----- results:")
        print("\n acc: ", acc)
        print("f1 macro: ", f1)
        print("auc macro: ", auc)
        print("confusion matrix:\n", conf_mat)


if __name__ == '__main__':
    linear_probe = LinearProbe("./data/ptbxl_dataset/", "./encoders/pre_train_model.h5",
                               do_inference=True)
    linear_probe.fit_eval()
