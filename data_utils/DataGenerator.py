import numpy as np
from data_utils.DataLoader import DataLoader


def transform_1d(sig):
    num_leads = sig.shape[-1]
    length_sig = sig.shape[0]
    variance_factor = 0.1

    for i in range(num_leads):
        sig_ld = sig[:, i]
        mask_length = np.random.randint(0, length_sig)
        mask_start = np.random.randint(0, length_sig - mask_length)
        if mask_length > 0:
            sig_ld[mask_start:(mask_start + mask_length)] = np.zeros([mask_length])
        gauss_noise = 10 * np.random.normal(0, variance_factor, size=length_sig)
        sig_ld += gauss_noise
        sig_ld = (sig_ld - np.min(sig_ld)) / (np.max(sig_ld) - np.min(sig_ld) + 1e-8)
        sig[:, i] = sig_ld
    return sig


class CLRGenerator(object):
    def __init__(self, data_path_1d, data_path_2d, use_data, batch_size):
        self.loader = DataLoader(data_path_1d=data_path_1d, data_path_2d=data_path_2d, use_data=use_data)
        self.loader.load()
        self.batch_size = batch_size
        self.segment_length_1d = self.loader.data_1d.shape[1]
        self.segment_length_2d = 300
        self.n_samples = self.loader.data_1d.shape[0]
        self.n_batches = self.n_samples // self.batch_size

    def get_batch(self):
        x_1d = np.empty(
            (2 * self.batch_size, self.segment_length_1d, 12),
            dtype=np.float32,
        )
        x_2d = np.empty(
            (2 * self.batch_size, self.segment_length_2d, self.segment_length_2d, 1),
            dtype=np.float32,
        )
        idx = np.arange(self.batch_size)
        data_1d, data_2d = self.loader.get_patients(self.batch_size)
        for i in range(self.batch_size):
            x_1d[idx[i]] = data_1d[i]
            x_1d[self.batch_size + idx[i]] = transform_1d(data_1d[i])
            x_2d[idx[i]] = data_2d[i]
        return [x_1d, x_2d], np.zeros([2 * self.batch_size, 256])

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_batches

    def __next__(self):
        return self.get_batch()
