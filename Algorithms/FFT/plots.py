# coding: utf-8
#This file generates the plots for the 1d fft algorithms


## TODO: update plots and verify
import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
# needed because of the number of points cairo can plot is limited
matplotlib.rcParams['backend'] = 'agg'

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy import fft
from pyfftw.interfaces import scipy_fftpack as pyfft

#plots pulseramp waev
def pulseramp():
    rate, mywave = wavfile.read('pulseramp.wav')
    plt.plot(mywave)
    plt.savefig('pulseramp.png')
    plt.clf()
    
#plots spectrum of sine wave
def sine_spec():
    samplerate = 44100 # 44100 samples per second
    freq = 1760 # We’re going to produce a 1760 Hz sine wave ...
    length = 2 # ... which will last for 2 seconds.
    stepsize = freq * 2*np.pi/samplerate
    sig = np.sin(np.arange(0, stepsize*length*samplerate, stepsize))     
    sig = np.float32(sig)
    fsig = pyfft.fft(sig)
    plt.plot(np.absolute(fsig))
    plt.savefig('sinespec.png')
    plt.clf()
    
#plots spectrum of tada.wav
def tada_spec():
    rate, sig = wavfile.read('tada.wav')
    sig = np.float32(sig)
    fsig = pyfft.fft(sig.T).T
    plt.plot(np.absolute(fsig))
    plt.savefig('tadaspec.png')
    plt.clf()

#plots left half of spectrum of tada.wav
def tada_spec_left():
    rate, sig = wavfile.read('tada.wav')
    sig = np.float32(sig)
    fsig = pyfft.fft(sig.T).T
    f = np.absolute(fsig)
    plt.plot(f[0:f.shape[0]/2,:])
    plt.savefig('tadaspec2.png')
    plt.clf()

def saw_spec():
    rate, data = wavfile.read('saw.wav')
    fsig = fft(data)
    f = np.absolute(fsig)
    plt.plot(f[0:f.shape[0]/2])
    plt.savefig('sawspec.png')
    plt.clf()

def down_saw_spec():
    rate, data = wavfile.read('down_saw.wav')
    fsig = fft(data)
    f = np.absolute(fsig)
    plt.plot(f[0:f.shape[0]/2])
    plt.savefig('sawspecdown.png')    
    plt.clf()
    
def down_saw_spec_correct():
    rate, in_sig = wavfile.read('saw.wav')
    old_rate = 44100
    new_rate = 22050
    in_sig = np.float32(in_sig)
    fin = pyfft.fft(in_sig)
    nsiz = np.floor(in_sig.size*new_rate/old_rate)
    nsizh = np.floor(nsiz/2)
    fout = np.zeros(nsiz)
    fout = fout + 0j
    fout[0:nsizh] = fin[0:nsizh]
    fout[nsiz-nsizh+1:] = np.conj(np.flipud(fout[1:nsizh]))
    f = np.absolute(fout)
    plt.plot(f[0:f.shape[0]/2])
    plt.savefig('sawdownspec.png')
    plt.clf()
    
if __name__ == "__main__":
    pulseramp()
    sine_spec()
    tada_spec()
    tada_spec_left()
    saw_spec()
    down_saw_spec()
    down_saw_spec_correct()
    
    