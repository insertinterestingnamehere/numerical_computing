import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
from matplotlib import pyplot as plt

import numpy as np
import scipy as sp
from scipy import misc
import pyfftw

def hamming(n):
    """
    Generate a hamming window of n points as a numpy array.
    """
    return 0.54 - 0.46 * np.cos(2 * np.pi / n * (np.arange(n) + 0.5))

def FFT2(A):
    B = pyfftw.interfaces.scipy_fftpack.fft(A, axis=0)
    return pyfftw.interfaces.scipy_fftpack.fft(B, axis=1)

def powerCepstrum():
    '''
    Generate plots of the steps in creating the power Cepstrum.
    '''
    d = np.linspace(0,4*np.pi, 200)
    s = np.sin(d)+.1*np.sin(12*d)+.1*np.cos(12*d)**2 + .1*np.sin(18*d)**2
    ax = plt.subplot(221)
    ax.plot(s)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.xlabel('Original Signal')
    ax = plt.subplot(222)
    fs = pyfftw.interfaces.scipy_fftpack.fft(s)
    ax.plot(fs[:len(fs)/2])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.xlabel('Fourier Transform')
    fs[np.abs(fs)<1e-100] = 1e-100
    lfs = np.log(np.abs(fs)**2)
    ax = plt.subplot(223)
    ax.plot(lfs[:len(lfs)/2])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.xlabel('Squared Log')
    pc = pyfftw.interfaces.scipy_fftpack.ifft(lfs)
    ax = plt.subplot(224)
    ax.plot(pc[1:len(pc)/2])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.xlabel('Power Cepstrum')
    plt.savefig('PowerCepstrum.pdf')
    plt.clf()

def hammingWindow():
    '''
    Generate plot of a Hamming window and windowed signal.
    '''
    d = np.linspace(0,4*np.pi, 200)
    h = 200*hamming(200)
    sig = 100*np.sin(3*d)
    windowed = sig*h/200
    ax = plt.subplot(131, aspect = 'equal')
    ax.plot(h)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.xlabel('Hamming Window Function')
    ax = plt.subplot(132, aspect = 'equal')
    ax.plot(sig)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_autoscale_on(True)
    plt.xlabel('Original Signal')
    ax = plt.subplot(133, aspect = 'equal')
    ax.plot(windowed)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.xlabel('Windowed Signal')
    plt.savefig('Hamming.pdf')
    plt.clf()

def plotFFT():
    L = misc.lena()
    E = plt.imread('ecoli.jpg')[:512, :512]
    FL = FFT2(L)
    FE = FFT2(E)
    FLs =  pyfftw.interfaces.scipy_fftpack.fftshift(FL)
    FEs =  pyfftw.interfaces.scipy_fftpack.fftshift(FE)
    mag1 = np.abs(FLs)
    mag1 *= 1000/mag1.max()
    mag1[mag1 > 1] = 1
    mag2 = np.abs(FEs)
    mag2 *= 1000/mag2.max()
    mag2[mag2 > 1] = 1
    ax = plt.subplot(221)
    ax.imshow(L, plt.cm.Greys_r)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.xlabel('Lena Image')
    ax = plt.subplot(222)
    ax.imshow(E, plt.cm.Greys_r)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.xlabel('E. Coli Image')
    ax = plt.subplot(223)
    ax.imshow(mag1, plt.cm.Blues)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.xlabel('FT of Lena')
    ax = plt.subplot(224)
    ax.imshow(mag2, plt.cm.Blues)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.xlabel('FT of E. Coli')
    plt.savefig('2dfft.pdf')
    plt.clf()

powerCepstrum()
hammingWindow()
plotFFT()
    
