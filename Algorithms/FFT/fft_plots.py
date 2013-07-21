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
    plt.close('all')
    rate, mywave = wavfile.read('pulseramp.wav')
    plt.plot(mywave)
    plt.savefig('pulseramp.pdf')
    
#plots spectrum of sine wave
def plot_sine_spec():
    plt.close('all')
    samplerate = 44100 # 44100 samples per second
    freq = 1760 # We’re going to produce a 1760 Hz sine wave ...
    length = 2 # ... which will last for 2 seconds.
    stepsize = freq*2*sp.pi/samplerate
    sig = sp.sin(sp.arange(0,stepsize*length*samplerate ,stepsize))     
    sig = sp.float32(sig)
    fsig = anfft.fft(sig)
    plt.plot(sp.absolute(fsig))
    plt.savefig('sinespec.pdf')
    
#plots spectrum of tada.wav
def plot_tada_spec():
    plt.close('all')
    rate, sig = wavfile.read('tada.wav')
    sig = sp.float32(sig)
    fsig = anfft.fft(sig.T).T
    plt.figure()
    plt.plot(sp.absolute(fsig))
    plt.savefig('tadaspec.pdf')

#plots left half of spectrum of tada.wav
def plot_tada_spec_left():
    plt.close('all')
    rate, sig = wavfile.read('tada.wav')
    sig = sp.float32(sig)
    fsig = anfft.fft(sig.T).T
    f = sp.absolute(fsig)
    plt.figure()
    plt.plot(f[0:f.shape[0]/2,:])
    plt.savefig('tadaspec2.pdf')
    
#plots Noisysignal1.wav
#TODO NEG    
def plot_noise():
    plt.close('all')
    rate, sig = wavfile.read('Noisysignal1.wav')
    plt.figure()
    plt.plot(sig)
    plt.show()
#    plt.savefig('noise.pdf')
    
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
#TODO NEG    
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
    
def plot_saw_spec():
    plt.close('all')
    rate, data = wavfile.read('saw.wav')
    fsig = sp.fft(data)
    f = sp.absolute(fsig)
    plt.figure()
    plt.plot(f[0:f.shape[0]/2])
    plt.savefig('sawspec.pdf')


def plot_down_saw_spec():
    plt.close('all')
    rate, data = wavfile.read('down_saw.wav')
    fsig = sp.fft(data)
    f = sp.absolute(fsig)
    plt.figure()
    plt.plot(f[0:f.shape[0]/2])
    plt.savefig('sawspecdown.pdf')    
    
def plot_down_saw_spec_correct():
    plt.close('all')
    rate, in_sig = wavfile.read('saw.wav')
    old_rate = 44100
    new_rate = 22050
    in_sig = sp.float32(in_sig)
    fin = anfft.fft(in_sig)
    nsiz = sp.floor(in_sig.size*new_rate/old_rate)
    nsizh = sp.floor(nsiz/2)
    fout = sp.zeros(nsiz)
    fout = fout + 0j
    fout[0:nsizh] = fin[0:nsizh]
    fout[nsiz-nsizh+1:] = sp.conj(sp.flipud(fout[1:nsizh]))
    f = sp.absolute(fout)
    plt.plot(f[0:f.shape[0]/2])
    plt.savefig('sawdownspec.pdf')
