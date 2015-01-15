import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
from matplotlib import pyplot as plt

import numpy as np
import scipy as sp
from scipy import misc
import pyfftw
from helperCode import hamming

def FFT2(A):
    B = pyfftw.interfaces.scipy_fftpack.fft(A, axis=0)
    return pyfftw.interfaces.scipy_fftpack.fft(B, axis=1)

def melScale(f):
    return 2595*np.log10(1+f/700)

def PowerCepstrum():
	'''
	Generate plots of the steps in creating the power Cepstrum.
	'''
	d = np.linspace(0,4*np.pi, 200)
	s = np.sin(d)+.1*np.sin(12*d)+.1*np.cos(12*d)**2 + .1*np.sin(18*d)**2
	plt.plot(s)
	plt.xticks([])
	plt.yticks([])
	plt.savefig('OriginalSignal.pdf')
	plt.close()
	
	fs = pyfftw.interfaces.scipy_fftpack.fft(s)
	fs2 = fs[:len(fs)/2]
	plt.plot(fs2)
	plt.xticks([])
	plt.yticks([])
	plt.savefig('FourierTransform.pdf')
	plt.close()
	
	fs[np.abs(fs)<1e-100] = 1e-100
	lfs = np.log(np.abs(fs)**2)
	lfs2 = lfs[:len(lfs)/2]
	plt.plot(lfs2)
	plt.xticks([])
	plt.yticks([])
	plt.savefig('SquaredLog.pdf')
	plt.close()
	
	pc = pyfftw.interfaces.scipy_fftpack.ifft(lfs)
	pc2 = pc[1:len(pc)/2]
	plt.plot(pc2)
	plt.xticks([])
	plt.yticks([])
	plt.savefig('PowerCepstrum.pdf')
	plt.close()
	
def hammingWindow():
	'''
	Generate plot of a Hamming window and windowed signal.
	'''
	d = np.linspace(0,4*np.pi, 200)
	h = 200*hamming(200)
	sig = 100*np.sin(3*d)
	windowed = sig*h/200
	plt.plot(h)
	plt.xticks([])
	plt.yticks([])
	plt.savefig('HammingWindowFunction.pdf')
	plt.close()
	
	plt.plot(sig)
	plt.xticks([])
	plt.yticks([])
	plt.savefig('Original.pdf')
	plt.close()
	
	plt.plot(windowed)
	plt.xticks([])
	plt.yticks([])
	plt.savefig('WindowedSignal.pdf')
	plt.close()
	
def plotMelScale():
	'''
	Generate plot of the mel scale and a traditional mel filterbank
	'''
	dom = np.linspace(0,10000, 1000)
	ms = melScale(dom)
	plt.subplot(211)
	plt.subplots_adjust(hspace=.5)
	plt.plot(dom, ms, 'm', lw = 5)
	plt.title('Mel Scale vs. Hz Scale')
	plt.xlim([0,10000])
	plt.ylim([0,4000])
	plt.grid(b=True, which='major', color='b', linestyle='--')
	plt.tick_params(labelsize=8)
	plt.xlabel('Herz scale')
	plt.ylabel('Mel scale')
	
	bins = 13
	hz_max = 8000
	mel_max = 2595*np.log10(1+hz_max/700.)
	mel_bins = np.linspace(0, mel_max, bins+2)
	hz_bins = 700*(10**(mel_bins/2595.)-1)
	l1 = np.zeros(hz_bins.shape)
	l2 = np.zeros(hz_bins.shape)
	l1[1::2] = 1
	l2[2:-1:2] = 1
	plt.subplot(212)
	plt.plot(hz_bins, l1, 'm', hz_bins, l2, 'm')
	plt.title('Mel Filterbank')
	plt.xlim([0,8000])
	plt.tick_params(labelsize=8)
	plt.xlabel('Hz scale')
	plt.ylabel('Magnitude')

	plt.savefig('MelScale.pdf')
	plt.close()
	
def ecoli():
	E = plt.imread('ecoli.jpg')
	FE = FFT2(E)
	FEs =  pyfftw.interfaces.scipy_fftpack.fftshift(FE)
	mag2 = np.abs(FEs)
	mag2 *= 1000/mag2.max()
	mag2[mag2 > 1] = 1
	mag2 = 255 - mag2
	sp.misc.imsave('ecoliFFT.pdf',mag2)



if __name__ == "__main__":
	PowerCepstrum()
	hammingWindow()
	plotMelScale()
	ecoli()