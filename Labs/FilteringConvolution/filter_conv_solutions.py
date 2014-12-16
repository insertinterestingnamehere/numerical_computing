import scipy as sp
from scipy.io import wavfile
from matplotlib import pyplot as plt

#==============================================================================
# PROBLEM 1
#==============================================================================
def prob1():
	rate,data = wavfile.read('Noisysignal2.wav')
	fsig = sp.fft(data,axis = 0)
	f = sp.absolute(fsig)
	plt.plot(f[0:f.shape[0]/2])
	for j in xrange(14020,50001):
		fsig[j]=0
		fsig[-j]=0

	newsig=sp.ifft(fsig)
	f = sp.absolute(fsig)
	plt.figure()
	plt.plot(f[0:f.shape[0]/2])
	plt.show()
	plt.close()
	newsig = sp.ifft(fsig).astype(float)
	scaled = sp.int16(newsig/sp.absolute(newsig).max() * 32767)
	wavfile.write('cleansig2.wav',rate,scaled)
#==============================================================================
# PROBLEM 3
#==============================================================================
def prob3():
	rate1,sig1 = wavfile.read('chopinw.wav')
	n = sig1.shape[0]
	rate2,sig2 = wavfile.read('balloon.wav')
	m = sig2.shape[0]
	sig1 = sp.append(sig1,sp.zeros((m,2)))
	sig2 = sp.append(sig2,sp.zeros((n,2)))
	f1 = sp.fft(sig1)
	f2 = sp.fft(sig2)
	out = sp.ifft((f1*f2))
	out = sp.real(out)
	scaled = sp.int16(out/sp.absolute(out).max() * 32767)
	wavfile.write('test.wav',rate1,scaled)
#==============================================================================
# PROBLEM 4
#==============================================================================
def prob4():
	samplerate = 22050
	noise = sp.int16(sp.random.randint(-32767,32767,samplerate*10)) # Create 10 seconds of mono white noise
	wavfile.write('white_noise.wav',22050,noise)
	f = sp.fft(sp.float32(noise))
	plt.plot(sp.absolute(f))
	plt.show()
#==============================================================================
# PROBLEM 5
#==============================================================================
def prob5():
	rate, sig = wavfile.read('tada.wav')
	sig = sp.float32(sig)
	noise = sp.float32(sp.random.randint(-32767,32767,sig.shape))
	out = sp.ifft(sp.fft(sig)*sp.fft(noise))
	out = sp.real(out)
	out = sp.int16(out/sp.absolute(out).max() * 32767)
	wavfile.write('white-conv.wav',rate,out)

