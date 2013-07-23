# coding: utf-8
#This file generates the plots for the 1d fft algorithms
# lab
import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import matplotlib.pyplot as plt
import scipy as sp
from scipy.io import wavfile
from scipy import fft
import anfft

#plots Noisysignal1.wav
def plot_noise():
    plt.close('all')
    rate, sig = wavfile.read('Noisysignal1.wav')
    plt.figure()
    plt.plot(sig)
    plt.savefig('noisy.pdf')
    
#plots lect half of spectrum of 
#Noisysignal1.wav    
def plot_noise_spec():
    plt.close('all')
    rate, sig = wavfile.read('Noisysignal1.wav')
    sig = sp.float32(sig)
    fsig = anfft.fft(sig.T).T
    f = sp.absolute(fsig)
    plt.figure()
    plt.plot(f[0:f.shape[0]/2])
    plt.savefig('noisyspec.pdf')
    
#plots cleaned noisy signal
def plot_cleaned_signal():
    plt.close('all')
    rate,data = wavfile.read('Noisysignal1.wav')
    fsig = sp.fft(data,axis = 0)
    for j in xrange(10000,20000):
        fsig[j]=0
        fsig[-j]=0

    newsig=sp.ifft(fsig)
    newsig = sp.real(newsig)
    newsig = sp.int16(newsig/sp.absolute(newsig).max() * 32767)    
    plt.figure()
    plt.plot(newsig)
    plt.savefig('Cleanedsignal.pdf')
    

