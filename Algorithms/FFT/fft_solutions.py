"""
Created on Wed May 22 11:55:48 2013

@author: Jeff Hendricks
"""

import scipy as sp
from scipy.fftpack import fft, ifft
from scipy.io import wavfile
from matplotlib import pyplot as plt
import anfft
#rate, data = wavfile.read('pulseramp.wav')
##plt.plot(data)
##plt.show()
#neg_data = data*-1
##wavfile.write('negpulseramp.wav',rate,neg_data)
#
##Here's one way of producing a a 1760Hz sine wave (which in musical terms is A6, the A
##between two and three octaves above middle C):
#samplerate=44100 # 44100 samples per second
#freq=1760 # We're going to produce a 1760 Hz sine wave ...
#length=2 # ... which will last for 2 seconds.
#stepsize=freq*2*sp.pi/samplerate
#sig=sp.sin(sp.arange(0,stepsize*length*samplerate,stepsize))
#scaled = sp.int16(sig/sp.absolute(sig).max() * 32767)
##wavfile.write('sinusound.wav',samplerate, scaled)
#
##PROBLEM 2
#samplerate=44100 # 44100 samples per second
#freq=60 # We're going to produce a 1760 Hz sine wave ...
#length=1 # ... which will last for 2 seconds.
#stepsize=freq*2*sp.pi/samplerate
#sig=sp.sin(sp.arange(0,stepsize*length*samplerate,stepsize))
#scaled = sp.int16(sig/sp.absolute(sig).max() * 32767)
##plt.plot(scaled)
##plt.show()
#
#
#fsig = sp.fft(sig)
##plt.plot(sp.absolute(fsig))
##plt.show()

#rate, sig = wavfile.read('tada2.wav')
#fsig = sp.fft(sig,axis = 0)
##plt.figure()
##plt.plot(sp.absolute(fsig))
##plt.show()
#
#plt.figure()
#f = sp.absolute(fsig)
#plt.plot(f[0:f.shape[0]/2,:])

#===============================================================================
# 
#PROBLEM 4
#rate,data = wavfile.read('Noisysignal2.wav')
#fsig = sp.fft(data,axis = 0)
#f = sp.absolute(fsig)
#plt.plot(f[0:f.shape[0]/2])
#for j in xrange(14020,50001):
#    fsig[j]=0
#    fsig[-j]=0
#
#newsig=sp.ifft(fsig)
#f = sp.absolute(fsig)
#plt.figure()
#plt.plot(f[0:f.shape[0]/2])
#plt.show()
#newsig = sp.ifft(fsig).astype(float)
#scaled = sp.int16(newsig/sp.absolute(newsig).max() * 32767)
#wavfile.write('cleansig2.wav',rate,scaled)


#===============================================================================

#rate, in_sig = wavfile.read('saw.wav')
#out_sig = in_sig[sp.arange(0,rate,2)]
##wavfile.write('down_saw.wav',rate/2,out_sig)
#
#old_rate = 44100
#new_rate = 22050
#fin = sp.fft(in_sig)
#nsiz = sp.floor(in_sig.size*new_rate/old_rate)
#nsizh = sp.floor(nsiz/2)
#fout = sp.zeros(nsiz)
#fout = fout + 0j
#fout[0:nsizh] = fin[0:nsizh]
#fout[nsiz-nsizh+1:] = sp.conj(sp.flipud(fout[1:nsizh]))
#out = sp.ifft(fout)
#out = sp.real(out)
#out = sp.int16(out/sp.absolute(out).max() * 32767)
#wavfile.write('down_saw2.wav',new_rate,out)

#===============================================================================
# Problem 6/7
#old_rate, in_sig = wavfile.read('saw.wav')
#new_rate = 11025
#fin = sp.fft(in_sig)
#nsiz = sp.floor(in_sig.size*new_rate/old_rate)
#nsizh = sp.floor(nsiz/2)
#fout = sp.zeros(nsiz) + 0j
#fout[0:nsizh] = fin[0:nsizh]
#fout[nsiz-nsizh+1:] = sp.conj(sp.flipud(fout[1:nsizh]))
#out = sp.ifft(fout)
#out = sp.real(out)
#out = sp.int16(out/sp.absolute(out).max() * 32767)
#wavfile.write('prob6_saw.wav',new_rate,out)
#===============================================================================

#===============================================================================
# Problem 8
#old_rate, in_sig = wavfile.read('tada2.wav')
#n = in_sig.shape[0]
#chan_num = in_sig.shape[1]
#new_sig = sp.zeros((2*n,chan_num),dtype = 'int16')
#new_sig[::2,:] = in_sig
#sig2 = sp.insert(in_sig,0,0,axis = 0)
#sig3 = sp.append(in_sig,[[0,0]],axis = 0)
#odds = .5*(sig2+sig3)
#new_sig[1::2,:] = odds[1:] 
#f = sp.fft(new_sig,axis = 0)
##wavfile.write('naive_up.wav',44100,new_sig)
#new_rate = 44100
#fin = sp.fft(in_sig, axis = 0)
#nsiz = sp.floor(in_sig.shape[0]*new_rate/old_rate)
#nsizh = sp.floor(nsiz/2)
#fout = sp.zeros((nsiz,in_sig.shape[1])) + 0j
#fout[0:nsizh] = fin[0:nsizh]
#fout[nsiz-nsizh+1:] = sp.conj(sp.flipud(fout[1:nsizh]))
#out = sp.ifft(fout, axis = 0)
#out = sp.real(out)
#out = sp.int16(out/sp.absolute(out).max() * 32767)
#plt.plot(out)
#plt.show()
#wavfile.write('upsample_b.wav',44100,out)
#===============================================================================
#FILTERING AND CONVOLUTION
#===============================================================================
#PROBLEM 10
#==============================================================================
# rate1,sig1 = wavfile.read('chopinw.wav')
# n = sig1.shape[0]
# rate2,sig2 = wavfile.read('balloon.wav')
# m = sig2.shape[0]
# sig1 = sp.append(sig1,sp.zeros((m,2)),axis = 0)
# sig2 = sp.append(sig2,sp.zeros((sig1.shape[0] - m,2)),axis = 0)
# f1 = anfft.fft(sig1.T).T
# f2 = anfft.fft(sig2.T).T
# out = anfft.ifft((f1*f2).T).T
# #f1 = sp.fft(sig1,axis = 0)
# #f2 = sp.fft(sig2,axis = 0)
# #out = sp.ifft((f1*f2),axis = 0)
# out = sp.real(out)
# scaled = sp.int16(out/sp.absolute(out).max() * 32767)
# wavfile.write('test.wav',44100,scaled)
#==============================================================================
#PROBLEM 11
samplerate = 22050
noise = sp.int16(sp.random.randint(-32767,32767,samplerate*10)) # Create 10 seconds of mono white noise
wavfile.write('white_noise.wav',22050,noise)
f = anfft.fft(sp.float32(noise))
plt.plot(sp.absolute(f)); plt.show()
#==============================================================================
#rate, sig = wavfile.read('tada-conv.wav')
#sig = sp.append(sig,sig)
#sig = sp.append(sig,sig)
#wavfile.write('test.wav',44100,sig)
#==============================================================================
#PROBLEM 12
#rate, sig = wavfile.read('sounds.wav')
#sig = sp.float32(sig)
#noise = sp.float32(sp.random.randint(-32767,32767,sig.shape))
#out = anfft.ifft(anfft.fft(sig.T)*anfft.fft(noise.T)).T
#out = sp.real(out)
#out = sp.int16(out/sp.absolute(out).max() * 32767)
#wavfile.write('sounds-conv.wav',rate,out)
#==============================================================================
#Naive FT
#def myft(f):
#    N = f.size
#    c = sp.zeros((N,), dtype=sp.complex128)
#    for k in xrange(0,N):
#        c[k] = (1/float(N))*(sp.exp(-2*pi*1j*k*sp.arange(N)/float(N))*f).sum()
##        c[k] = (f * exp(-2*pi*1j*k*np.arange(N)/N)).sum()
#    return c
#==============================================================================
#pianoclip problem
#rate, sig = wavfile.read('pianoclip.wav')
#sig = sp.float32(sig)
#fsig = anfft.fft(sig.T).T
#f = sp.absolute(fsig)
#plt.plot(f[0:f.shape[0]/2]); plt.show()
##spikes at 3060 3640 4110 ...
#mult = rate/float(sig.shape[0])