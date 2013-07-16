# Solutions to problem 1
import scipy as sp
import numpy as sp
from matplotlib import pyplot as plt
from scipy import signal

def getFrame(m):
    coeffs = []
    k_max = int(sp.pi*2**(m+1))
    for k in xrange(k_max):
        coeffs.append(-2**m*(sp.cos((k+1)*2**(-m)) - sp.cos(k*2**(-m))))
    coeffs.append(-2**m*(1-sp.cos(k_max*2**(-m))))
    return sp.array(coeffs)

# frame_4 = getFrame(4)
# frame_6 = getFrame(6)
# frame_8 = getFrame(8)

# Here's how to plot each one:
# plt.plot([x*2*sp.pi/len(frame_6) for x in range(len(frame_6))],frame_6,drawstyle='steps')

# Solutions to problem 2

def getDetail(m):
    coeffs = []
    k_max = int(sp.pi*2**(m+1))
    for k in xrange(k_max):
        coeffs.append(-2**m*(2*sp.cos((2*k+1)*2**(-m-1)) - sp.cos((k+1)*2**(-m)) - sp.cos(k*2**(-m))))
    if (2*sp.pi < (2*k_max+1)*2**(-m-1)):
        coeffs.append(-2**m*(1-sp.cos(k_max*2**(-m))))
    else:
        coeffs.append(-2**m*(2*sp.cos((2*k_max+1)*2**(-m-1)) - 1 - sp.cos(k*2**(-m))))
    return sp.array(coeffs)

# detail = getDetail(4)

# Here's how to plot the details
#b = []
#for i in detail:
#    b.extend([i,-i])
#plt.plot([x*2*sp.pi/len(b) for x in range(len(b))],b,drawstyle='steps')

# Here's how to calculate the frame for m=5
#details = getDetail(4)
#frame = getFrame(4)
#frame_5 = sp.zeros(2*len(details))
#for i in range(2*len(details)):
#    frame_5[i] = frame[i/2] + (-1)**i*details[i/2]

''' Notes: these methods use the fftconvolve function from sp.signal.
	I have found that the default, mode='full', is the best setting.
	However, the convolution outputs a vector that is one entry too
	long, and so either the first or last element should be omitted.
'''

''' Single level wavelet decomposition.
	of the given signal wrt the given low-pass and hi-pass filters.
	Parameters: signal is a 1D array. lo_d and hi_d are 1D arrays, the 
	low-pass and high-pass decomposition filters, respectively. 
	Returns: list of two 1D arrays, the approximation and detail
	coefficients, respectively.
	'''
def dwt_pass(signal, lo_d, hi_d):
	a = sp.signal.fftconvolve(signal, lo_d)[1:]
	d = sp.signal.fftconvolve(signal, hi_d)[1:]
	return [a[::2],d[::2]]

''' Full wavelet decomposition.
	Parameters: signal is a 1D array to be decomposed. lo_d and hi_d
	are the decomposition filters.
	Returns: a list of 1D arrays starting with the final level of 
	approximation coefficients followed by the detail coefficients 
	working back up to the first level.
	'''
def dwt(signal, lo_d, hi_d):
	length = len(signal)
	result = []
	sig = signal
	while(length >= len(lo_d)):
		sig,details = dwt_pass(sig,lo_d,hi_d)
		result.append(details)
		length/=2
	result.append(sig)
	result.reverse()
	return result

''' Single level wavelet reconstruction.
	Parameters: coeffs is a list of two 1D arrays, the approximation and 
	detail coefficients, respectively. lo_r and hi_r are the low-pass and
	high-pass reconstruction filters, respectively. 
	returns: a 1D array, the reconstructed signal
	'''
def idwt_pass(coeffs,lo_r,hi_r):
	up1 = sp.zeros(2*len(coeffs[0]))
	up2 = sp.zeros(2*len(coeffs[1]))
	up1[::2], up2[::2] = coeffs 
	return sp.signal.fftconvolve(up1,lo_r)[:-1] + sp.signal.fftconvolve(up2,hi_r)[:-1]

''' Full wavelet reconstruction.
	Parameters: coeffs is a list of 1D arrays starting first with the last level of 
	approximation coefficients, then followed by the detail coefficients working back
	up to level 1. lo_r and hi_r are the reconstruction filters, as usual.
	Returns: a 1D array, the reconstructed signal
	'''
def idwt(coeffs,lo_r,hi_r):
	result = coeffs[0]
	for i in xrange(len(coeffs)-1):
		args = [result,coeffs[i+1]]
		result = idwt_pass(args,lo_r,hi_r)
	return result

