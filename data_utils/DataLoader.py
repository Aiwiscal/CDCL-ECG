import numpy as np


class DataLoader(object):
    def __init__(self, data_path_1d, data_path_2d, use_data):
        self.data_path_1d = data_path_1d
        self.data_path_2d = data_path_2d
        self.use_data = use_data
        self.data_1d = None
        self.data_2d = None
        self.patient_ids = None
        self.patient_num = -1

    def load(self):
        if self.use_data == "train":
            self.data_1d = np.load(self.data_path_1d + "X_train.npy")
            self.data_2d = np.load(self.data_path_2d + "X_train_2d.npy")
        elif self.use_data == "val":
            self.data_1d = np.load(self.data_path_1d + "X_val.npy")
            self.data_2d = np.load(self.data_path_2d + "X_val_2d.npy")
        else:
            print("unknown dataset symbol : %s !" % self.use_data)
            exit(1)

        self.patient_num = self.data_1d.shape[0]
        self.patient_ids = np.arange(self.patient_num)

    def get_patients(self, n):
        patient_ids = np.random.choice(self.patient_ids, n, replace=False)
        patient_data_1d_list = list()
        patient_data_2d_list = list()
        for p_id in patient_ids:
            p_data_1d = self.data_1d[[p_id]]
            p_data_1d = (p_data_1d - np.mean(p_data_1d)) / (np.std(p_data_1d) + 1e-6)
            p_data_2d = self.data_2d[[p_id]]
            patient_data_1d_list.append(p_data_1d)
            patient_data_2d_list.append(p_data_2d)
        return np.concatenate(patient_data_1d_list, axis=0), np.concatenate(patient_data_2d_list, axis=0)
