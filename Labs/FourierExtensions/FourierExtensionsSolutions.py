'''
Solutions for the Fourier Extensions Lab of Volume 1
'''

import numpy as np
import pyfftw
from matplotlib import pyplot as plt
from helperCode import hamming, melfb

def powerCepstrum(f):
    '''
    Compute the power Cepstrum of the input signal f.
    Inputs:
        f -- one-dimensional array
    Returns:
        one-dimensional array, the power Cepstrum of f
    '''
    s = pyfftw.interfaces.scipy_fftpack.fft(f)
    s[np.abs(s)<1e-100] = 1e-100
    ls = np.log(np.abs(s)**2)
    print ls[:6]
    ils = pyfftw.interfaces.scipy_fftpack.ifft(ls)
    print ils[:6]
    return np.abs(ils)**2

def window(f):
    '''
    Break up a signal into frames, and multiply each frame with a Hamming window.
    Create 198 frames of length 1323, with adjacent frames overlapping by 882 values.
    Inputs:
        f -- array of shape (88200,)
    Returns:
        List of length 198, containing arrays of shape (1323,), ordered
        according to the position of the frames relative to f.
    '''
    window = hamming(1323)
    frames = []
    for i in xrange(198):
        frame = f[i*441: i*441 + 1323]*window
        frames.append(frame)
    return frames

def powerSpectrum(f):
    '''
    Perform pre-emphasis and compute the power Spectrum of an array.
    Inputs:
        f -- array of shape (1323,)
    Returns:
        array of shape (1025,), the power Spectrum of f
    '''
    f[1:] -= f[:-1] * .95
    return np.abs(pyfftw.interfaces.scipy_fftpack.fft(f, 2048)[:1025]) ** 2
    
def extract(x):
    '''
    Extract MFCC coefficients of the sound signal x in numpy array format.
    Inputs:
        x -- array of shape (88200,)
    Returns:
        array of shape (198,10), the MFCCs of x.
    '''
    window = hamming(1323)
    feature = []
    for i in xrange(198):
        # Windowing
        frame = x[i*441: i*441 + 1323]*window
        # Pre-emphasis
        frame[1:] -= frame[:-1] * .95
        # Power spectrum
        X = np.abs(pyfftw.interfaces.scipy_fftpack.fft(frame, n=2048)[:1025]) ** 2
        X[X < 1e-100] = 1e-100  # Avoid zero
        # Mel filtering, logarithm, DCT
        M = melfb(40, 2048, 44100)
        X = 0.25*pyfftw.interfaces.scipy_fftpack.dct(np.log(np.dot(M,X)))[1:11]
        feature.append(X)
    feature = np.row_stack(feature)
    return feature

def fft2(A):
    '''
    Calculate the fourier transform of A.
    Inputs:
        A -- array of shape (m,n)
    Returns:
        Array of shape (m,n) giving the fourier transform.
    '''
    B = pyfftw.interfaces.scipy_fftpack.fft(A, axis=0)
    return pyfftw.interfaces.scipy_fftpack.fft(B, axis=1)

def ifft2(A):
    '''
    Calculate the inverse fourier transform of A.
    Inputs:
        A -- array of shape (m,n)
    Returns:
        Array of shape (m,n) giving the inverse fourier transform.
    ''' 
    B = pyfftw.interfaces.scipy_fftpack.ifft(A, axis=1)
    return pyfftw.interfaces.scipy_fftpack.ifft(B, axis=0)

