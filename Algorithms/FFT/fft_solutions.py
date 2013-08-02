import scipy as sp
from scipy.fftpack import fft, ifft
from scipy.io import wavfile
from matplotlib import pyplot as plt
import anfft

#==============================================================================
# PROBLEM 1
#==============================================================================
#samplerate=44100 # 44100 samples per second
#freq=60   #60Hz sine wave
#length=1 # ... which will last for 1 seconds.
#stepsize=freq*2*sp.pi/samplerate
#sig=sp.sin(sp.arange(0,stepsize*length*samplerate,stepsize))
#scaled = sp.int16(sig/sp.absolute(sig).max() * 32767)
#plt.plot(scaled)
#plt.show()
#wavfile.write('sine.wav',samplerate,scaled)
#==============================================================================
# PROBLEM 2 - Naive FT
#==============================================================================
#def myft(f):
#    N = f.size
#    c = sp.zeros((N,), dtype=sp.complex128)
#    for k in xrange(0,N):
#        c[k] = (1/float(N))*(sp.exp(-2*pi*1j*k*sp.arange(N)/float(N))*f).sum()
#        c[k] = (f * exp(-2*pi*1j*k*np.arange(N)/N)).sum()
#    return c
#==============================================================================
# PROBLEM 3
#==============================================================================
#rate, sig = wavfile.read('pianoclip.wav')
#sig = sp.float32(sig)
#fsig = anfft.fft(sig.T).T
#f = sp.absolute(fsig)
#plt.plot(f[0:f.shape[0]/2])
#plt.show() #spike at about 3075
#freq = rate/float(sig.shape[0]) * 3075
#note is approximately an f#5
#==============================================================================
# Problem 4
#==============================================================================
#old_rate, in_sig = wavfile.read('saw.wav')
#new_rate = 11025
#fin = anfft.fft(sp.float32(in_sig))
#nsiz = sp.floor(in_sig.size*new_rate/old_rate)
#nsizh = sp.floor(nsiz/2)
#fout = sp.zeros(nsiz) + 0j
#fout[0:nsizh] = fin[0:nsizh]
#fout[nsiz-nsizh+1:] = sp.conj(sp.flipud(fout[1:nsizh]))
#out = sp.ifft(fout)
#out = sp.real(out)
#out = sp.int16(out/sp.absolute(out).max() * 32767)
#wavfile.write('prob4_saw.wav',new_rate,out)
#===============================================================================
# Problem 5
#==============================================================================
#old_rate, in_sig = wavfile.read('saw.wav')
#new_rate = 36000
#fin = anfft.fft(sp.float32(in_sig))
#nsiz = sp.floor(in_sig.size*new_rate/old_rate)
#nsizh = sp.floor(nsiz/2)
#fout = sp.zeros(nsiz) + 0j
#fout[0:nsizh] = fin[0:nsizh]
#fout[nsiz-nsizh+1:] = sp.conj(sp.flipud(fout[1:nsizh]))
#out = anfft.ifft(fout)
#out = sp.real(out)
#out = sp.int16(out/sp.absolute(out).max() * 32767)
#wavfile.write('prob5_saw.wav',new_rate,out)
#===============================================================================
