import scipy as sp
from scipy.fftpack import fft, ifft
from scipy.io import wavfile
from matplotlib import pyplot as plt
import anfft

#==============================================================================
# PROBLEM 1
#==============================================================================
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
#==============================================================================
# PROBLEM 3
#==============================================================================
#rate1,sig1 = wavfile.read('chopinw.wav')
#n = sig1.shape[0]
#rate2,sig2 = wavfile.read('balloon.wav')
#m = sig2.shape[0]
#sig1 = sp.append(sig1,sp.zeros((m,2)),axis = 0)
#sig2 = sp.append(sig2,sp.zeros((sig1.shape[0] - m,2)),axis = 0)
#f1 = anfft.fft(sig1.T).T
#f2 = anfft.fft(sig2.T).T
#out = anfft.ifft((f1*f2).T).T
#out = sp.real(out)
#scaled = sp.int16(out/sp.absolute(out).max() * 32767)
#wavfile.write('test.wav',44100,scaled)
#==============================================================================
# PROBLEM 4
#==============================================================================
#samplerate = 22050
#noise = sp.int16(sp.random.randint(-32767,32767,samplerate*10)) # Create 10 seconds of mono white noise
#wavfile.write('white_noise.wav',22050,noise)
#f = anfft.fft(sp.float32(noise))
#plt.plot(sp.absolute(f))
#plt.show()
#==============================================================================
# PROBLEM 5
#==============================================================================
rate, sig = wavfile.read('tada.wav')
sig = sp.float32(sig)
noise = sp.float32(sp.random.randint(-32767,32767,sig.shape))
out = anfft.ifft(anfft.fft(sig.T)*anfft.fft(noise.T)).T
out = sp.real(out)
out = sp.int16(out/sp.absolute(out).max() * 32767)
wavfile.write('white-conv.wav',rate,out)

