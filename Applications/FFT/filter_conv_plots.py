# coding: utf-8
#This file generates the plots for the 1d fft algorithms
# lab
import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import matplotlib.pyplot as plt

# Switch backends to render PNG images.
plt.switch_backend("Agg")

import scipy as sp
from scipy.io import wavfile
from scipy import fftpack as ft

#plots Noisysignal1.wav
def plot_noise():
    plt.close('all')
    rate, sig = wavfile.read('Noisysignal1.wav')
    plt.figure()
    plt.plot(sig[0:sig.shape[0]/2])
    plt.savefig('noisy.png')
    
#plots lect half of spectrum of 
#Noisysignal1.wav    
def plot_noise_spec():
    plt.close('all')
    rate, sig = wavfile.read('Noisysignal1.wav')
    sig = sp.float32(sig)
    fsig = ft.fft(sig.T).T
    f = sp.absolute(fsig)
    plt.figure()
    plt.plot(f[0:f.shape[0]/2])
    plt.savefig('noisyspec.png')
    
#plots cleaned noisy signal
def plot_cleaned_signal():
    plt.close('all')
    rate,data = wavfile.read('Noisysignal1.wav')
    fsig = ft.fft(data,axis = 0)
    for j in xrange(10000,20000):
        fsig[j]=0
        fsig[-j]=0

    newsig = ft.ifft(fsig)
    newsig = sp.real(newsig)
    newsig = sp.int16(newsig/sp.absolute(newsig).max() * 32767)    
    plt.figure()
    plt.plot(newsig[0:newsig.shape[0]/2])
    plt.savefig('Cleanedsignal.png')

if __name__ == '__main__':
    plot_noise()
    plot_noise_spec()
    plot_cleaned_signal()
