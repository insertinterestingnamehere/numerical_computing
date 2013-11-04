'''
This file contains helper code for use in the Fourier Extensions lab of Volume 1.
'''

import numpy as np

def hamming(n):
    """
    Generate a hamming window of n points as a numpy array.
    """
    return 0.54 - 0.46 * np.cos(2 * np.pi / n * (np.arange(n) + 0.5))

def melfb(p, n, fs):
    """
    Return a Mel filterbank matrix as a numpy array.
    Inputs:
        p:  number of filters in the filterbank
        n:  length of fft
        fs: sample rate in Hz
    Returns:
        M: a Mel filterbank matrix satisfying the inputs
    Ref. http://www.ifp.illinois.edu/~minhdo/teaching/speaker_recognition/code/melfb.m
    """
    f0 = 700.0 / fs
    fn2 = int(np.floor(n/2))
    lr = np.log(1 + 0.5/f0) / (p+1)
    CF = fs * f0 * (np.exp(np.arange(1, p+1) * lr) - 1)
    bl = n * f0 * (np.exp(np.array([0, 1, p, p+1]) * lr) - 1)
    b1 = int(np.floor(bl[0])) + 1
    b2 = int(np.ceil(bl[1]))
    b3 = int(np.floor(bl[2]))
    b4 = min(fn2, int(np.ceil(bl[3]))) - 1
    pf = np.log(1 + np.arange(b1,b4+1) / f0 / n) / lr
    fp = np.floor(pf)
    pm = pf - fp
    M = np.zeros((p, 1+fn2))
    for c in np.arange(b2-1,b4):
        r = fp[c] - 1
        M[r,c+1] += 2 * (1 - pm[c])
    for c in np.arange(b3):
        r = fp[c]
        M[r,c+1] += 2 * pm[c]
    return M
