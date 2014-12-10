# coding: utf-8
#This file generates the plots for the 1d fft algorithms
# lab
import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import matplotlib.pyplot as plt

# Switch backends to render PNG images.
plt.switch_backend("Agg")

import numpy as np
from scipy.io import wavfile
from scipy import fftpack as ft

#plots Noisysignal1.wav
def noise():
    plt.close('all')
    rate, sig = wavfile.read('Noisysignal1.wav')
    plt.plot(sig[0:sig.shape[0]/2])
    plt.savefig('noisy.png')
    plt.clf()
    
#plots lect half of spectrum of 
#Noisysignal1.wav    
def noise_spec():
    rate, sig = wavfile.read('Noisysignal1.wav')
    sig = sig.astype('float32')
    fsig = ft.fft(sig.T).T
    f =  np.absolute(fsig)
    plt.plot(f[0:f.shape[0]/2])
    plt.savefig('noisyspec.png')
    plt.clf()
    
#plots cleaned noisy signal
def cleaned_signal():
    rate,data = wavfile.read('Noisysignal1.wav')
    fsig = ft.fft(data,axis = 0)
    for j in xrange(10000,20000):
        fsig[j]=0
        fsig[-j]=0

    newsig = ft.ifft(fsig)
    newsig = newsig.real
    newsig = (newsig/np.absolute(newsig).max()*32767).astype('int16')
    plt.plot(newsig[0:newsig.shape[0]/2])
    plt.savefig('Cleanedsignal.png')
    plt.clf()

if __name__ == '__main__':
    noise()
    noise_spec()
    cleaned_signal()
