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

#plots pulseramp waev
def plot_pulse_ramp():
    rate, mywave = wavfile.read('pulseramp.wav')
    plt.plot(mywave)
    plt.savefig('pulseramp.pdf')
    
#plots spectrum of sine wave
def plot_sine_spec():
    samplerate=44100 # 44100 samples per second
    freq=1760 # We’re going to produce a 1760 Hz sine wave ...
    length=2 # ... which will last for 2 seconds.
    stepsize=freq*2*sp.pi/samplerate
    sig=sp.sin(sp.arange(0,stepsize*length*samplerate ,stepsize))     
    sig = sp.float32(sig)
    fsig = anfft.fft(sig)
    plt.plot(sp.absolute(fsig))
    plt.savefig('sinespec.pdf')
    
#plots spectrum of tada.wav
def plot_tada_spec():
    rate, sig = wavfile.read('tada.wav')
    sig = sp.float32(sig)
    fsig = anfft.fft(sig.T).T
    plt.figure()
    plt.plot(sp.absolute(fsig))
    plt.savefig('tadaspec.pdf')

#plots left half of spectrum of tada.wav
def plot_tada_spec_left():
    rate, sig = wavfile.read('tada.wav')
    sig = sp.float32(sig)
    fsig = anfft.fft(sig.T).T
    f = sp.absolute(fsig)
    plt.figure()
    plt.plot(f[0:f.shape[0]/2,:])
    plt.savefig('tadaspec2.pdf')
    
#plots Noisysignal1.wav
def plot_noise():
    rate, sig = wavfile.read('Noisysignal1.wav')
    plt.figure()
    plt.plot(sig)
    plt.savefig('noise.pdf')