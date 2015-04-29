import scipy as sp
import numpy as np
from scipy.fftpack import fft, ifft
from scipy.io import wavfile
from matplotlib import pyplot as plt
from pyfftw.interfaces import scipy_fftpack as fftw


#=============================================================================
# PROBLEM 1 - Plotting signals, create simple sine signal
#=============================================================================
def plot_signal(filename='pulseramp.wav', verbose=False):
    """Plots the signal of any given .wav file.
    
    Parameters
    ----------
    filename : string, optional
        The name of the .wav sound file to be plotted.
        Defaults to 'pulseramp.wav'.
    verbose : boolean, optional
        If True, prints out basic information about the signal.
        Defaults to False.
    
    Returns
    -------
    None
    """
    
    rate, mywave = wavfile.read(filename)
    if verbose:
        print "file:\t" + filename
        print "rate:\t" + str(rate)
        print "length:\t" + str(len(mywave))
    plt.plot(mywave)
    plt.title(filename)
    plt.show()

def prob1(freq=60, length=1):
    """Generates a sine wave, saves it as a .wav file, and uses plot_signal()
        to plot the signal.
    
    Parameters
    ----------
    freq : integer, optional
        The fequency of the sine wave. Defaults to 60.
    length : integer, optional
        The number of seconds the sine wave lasts. Defaults to 1.
    
    Returns
    -------
    None
    """
    
    samplerate = 44100
    stepsize = freq*2*sp.pi/samplerate
    signal = sp.sin(sp.arange(0, stepsize*length*samplerate, stepsize))
    scaled_signal = sp.int16(signal/sp.absolute(signal).max() * 32767)
    wavfile.write('problem1.wav', samplerate, scaled_signal)
    plot_signal('problem1.wav')

#=============================================================================
# PROBLEM 2 - Naive DFT
#=============================================================================
def prob2(vec, verbose=False):
    """A naive implementation of the Discrete Fourier Transform.
    
    Parameters
    ----------
    vec : array_like
        The 1 x N-1 vector [f(0),f(1),...,f(N-1)].
    verbose : boolean, optional
        If True, prints out whether or not the DFT was successful,
        comparing with scipy.fft(). Defaults to False.
    
    Returns
    -------
    c : array_like
        The 1 x N-1 vector of the DFT of 'vec'.
    """
    
    vec = sp.array(vec, dtype=sp.complex128)
    N = len(vec)
    c = sp.zeros(N, dtype=sp.complex128)
    for k in xrange(N):
        c[k] = 1./N*sp.sum(sp.exp((-2*sp.pi*1j*k*sp.arange(N))/N)*vec)
        #c[k] = (vec * sp.exp(-2*sp.pi*1j*k*sp.arange(N)/N)).sum()
    if verbose:
        if sp.allclose(sp.fft(vec)/float(N), c): print "Success!"
        else: print "Failure!"
    return c

#=============================================================================
# PROBLEM 3
#=============================================================================
def prob3(filename='pianoclip.wav'):
    """Plots the spectrum of a given .wav file, then calculates the location
    and value of the largest spike. For the default value, the exact value is
    742.281519994 Hz (f#5 + 5 cents)
    
    Parameters
    ----------
    filename: string, optional
        The name of the .wav sound file to be examined.
        Defaults to 'pianoclip.wav'.

    Returns
    -------
    None
    """
    plot_signal(filename)
    rate, signal = wavfile.read(filename)
    signal = sp.float32(signal)
    fsignal = sp.absolute(fftw.fft(signal.T).T)
    # Use if scipy_fftpack is unavailable
    #fsignal = sp.absolute(sp.fft(signal, axis=0))
    plt.plot(fsignal[0:fsignal.shape[0]/2])
    plt.title("Spectrum of " + filename)
    plt.show()
    loc = fsignal[1:].argmax()
    val = fsignal[1:].max()
    print "\nSpike location:\t" + str(loc)
    print "Spike value:\t" + str(val)
    print "Hz:\t\t" + str(float(loc*rate)/signal.shape[0])

#==============================================================================
# Problem 4
#==============================================================================
def prob4(filename='saw.wav', new_rate = 11025, outfile='prob4.wav'):
    """Down-samples a given .wav file to a new rate and saves the resulting
    signal as another .wav file.
    
    Parameters
    ----------
    filename : string, optional
        The name of the .wav sound file to be down-sampled.
        Defaults to 'saw.wav'.
    new_rate : integer, optional
        The down-sampled rate. Defaults to 11025.
    outfile : string, optional
        The name of the new file. Defaults to prob4.wav.

    Returns
    -------
    None
    """
    old_rate, in_sig = wavfile.read(filename)
    fin = fftw.fft(sp.float32(in_sig))
    # Use if scipy_fftpack is unavailable
    # fin = sp.fft(sp.float32(in_sig))
    nsiz = sp.floor(in_sig.size * new_rate / old_rate)
    nsizh = sp.floor(nsiz / 2)
    fout = sp.zeros(nsiz) + 0j
    fout[0:nsizh] = fin[0:nsizh]
    fout[nsiz-nsizh+1:] = sp.conj(sp.flipud(fout[1:nsizh]))
    out = sp.real(sp.ifft(fout))
    out = sp.int16(out/sp.absolute(out).max() * 32767)
    plot_signal(filename)
    wavfile.write('prob4.wav',new_rate,out)
    print ""; plot_signal('prob4.wav')

#===============================================================================
# Problem 5
#==============================================================================
def prob5():
    """Try changing the sampling rate of saw.wav to something other than an
    integer factor (36000 Hz).
    """
    prob4('saw.wav', 36000, 'prob5.wav')
#===============================================================================
