import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from models.cnn_net import build_net
from tqdm import tqdm


train_x = np.load("./data/ptbxl_dataset/X_train.npy")
train_y = np.load("./data/ptbxl_dataset/y_train.npy")


val_x = np.load("./data/ptbxl_dataset/X_val.npy")
val_y = np.load("./data/ptbxl_dataset/y_val.npy")

input_size = 2048
num_classes = 5   # for ptbxl
# num_classes = 9 # for cpsc2018
train_y = to_categorical(train_y, num_classes=num_classes)
val_y = to_categorical(val_y, num_classes=num_classes)

keras.backend.clear_session()
tf.compat.v1.reset_default_graph()
for i in tqdm(range(train_x.shape[0])):
    train_x[i] = (train_x[i] - np.mean(train_x[i])) / (np.std(train_x[i]) + 1e-6)
for i in tqdm(range(val_x.shape[0])):
    val_x[i] = (val_x[i] - np.mean(val_x[i])) / (np.std(val_x[i]) + 1e-6)

inputs, outputs, _ = build_net([input_size, 12], 16, num_classes)

model = Model(inputs=inputs, outputs=outputs)
model.summary()

opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(x=train_x, y=train_y, validation_data=(val_x, val_y), batch_size=128, epochs=100, verbose=1)
del train_x, train_y

test_x = np.load("./data/ptbxl_dataset/X_test.npy")
test_y = np.load("./data/ptbxl_dataset/y_test.npy")
for i in tqdm(range(test_x.shape[0])):
    test_x[i] = (test_x[i] - np.mean(test_x[i])) / (np.std(test_x[i]) + 1e-6)

test_y_ = model.predict(test_x, verbose=1)
y = test_y
y_ = np.argmax(test_y_, axis=1)
conf_mat = confusion_matrix(y, y_)
acc = accuracy_score(y, y_)
f1 = f1_score(y, y_, average="macro")

y_one_hot = to_categorical(y, num_classes=num_classes)
y_one_hot_ = test_y_
auc = roc_auc_score(y_one_hot, y_one_hot_, average="macro")
print("\n acc: ", acc)
print("f1 macro: ", f1)
print("auc macro: ", auc)
print("confusion matrix:\n", conf_mat)
