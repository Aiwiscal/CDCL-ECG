# -*- coding: utf-8 -*-
# @Author   : WenHan
import numpy as np
import pywt


def wt_filter(raw_signal, w="db6"):
    signal = np.zeros(raw_signal.shape)
    for i in range(raw_signal.shape[-1]):
        sig = raw_signal[:, i]
        # perform wavelet decomposition
        coefficients = pywt.wavedec(sig, w, level=7)
        # remove targeted coefficients
        coefficients[7] = np.zeros(coefficients[7].shape)
        coefficients[0] = np.zeros(coefficients[0].shape)
        # reconstruct signal
        sig = pywt.waverec(coefficients, w)
        # ensure that the length of reconstructed signal is consistent with the raw signal
        if len(sig) > signal.shape[0]:
            sig = sig[:signal.shape[0]]
        elif len(sig) < signal.shape[0]:
            sig = list(sig) + [sig[-1]] * int(signal.shape[0] - len(sig))
        else:
            pass
        signal[:, i] = sig
    return signal


if __name__ == '__main__':
    x = np.zeros([2048, 12])
    y = wt_filter(x)
    print(y.shape)
