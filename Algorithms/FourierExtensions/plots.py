import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
from matplotlib import pyplot as plt

import numpy as np
import scipy as sp
from scipy import misc
import pyfftw
from helperCode import hamming


def FFT2(A):
    B = pyfftw.interfaces.scipy_fftpack.fft(A, axis=0)
    return pyfftw.interfaces.scipy_fftpack.fft(B, axis=1)


def melScale(f):
    return 2595*np.log10(1+f/700)


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


def plotMelScale():
    '''
    Generate plot of the mel scale and a traditional mel filterbank.
    '''
    dom = np.linspace(0,10000, 1000)
    ms = melScale(dom)
    plt.subplot(211)
    plt.subplots_adjust(hspace=.5)
    plt.plot(dom, ms, 'm', lw = 5)
    plt.title('Mel Scale vs. Hz Scale')
    plt.xlim([0,10000])
    plt.ylim([0,4000])
    plt.grid(b=True, which='major', color='b', linestyle='--')
    plt.tick_params(labelsize=8)
    plt.xlabel('Herz scale')
    plt.ylabel('Mel scale')
    bins = 13
    hz_max = 8000
    mel_max = 2595*np.log10(1+hz_max/700.)
    mel_bins = np.linspace(0, mel_max, bins+2)
    hz_bins = 700*(10**(mel_bins/2595.)-1)
    l1 = np.zeros(hz_bins.shape)
    l2 = np.zeros(hz_bins.shape)
    l1[1::2] = 1
    l2[2:-1:2] = 1
    plt.subplot(212)
    plt.plot(hz_bins, l1, 'm', hz_bins, l2, 'm')
    plt.title('Mel Filterbank')
    plt.xlim([0,8000])
    plt.tick_params(labelsize=8)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.savefig('melScale.pdf')
    plt.clf()


def plotFFT():
    '''
    Generate plots of the 2D FFT of the Lena image and another image.
    '''
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


if __name__ == "__main__":
    powerCepstrum()
    hammingWindow()
    plotFFT()
    plotMelScale()
    
