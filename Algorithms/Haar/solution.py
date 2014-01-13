# Solutions to problem 1
import scipy as sp
import numpy as sp
from matplotlib import pyplot as plt
from scipy.signal import fftconvolve

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

def dwt(f, lo, hi):
    '''
    Compute the discrete wavelet transform of f with respect to 
    the wavelet filters lo and hi.
    Inputs:
        f -- numpy array corresponding to the signal
        lo -- numpy array giving the lo-pass filter
        hi -- numpy array giving the hi-pass filter
    Returns:
        list of the form [A, D1, D2, ..., Dn] where each entry
        is a numpy array. These are the approximation frame (A)
        and the detail coefficients.
    '''
    ans = []
    frame = f
    while len(frame) >= len(lo):
        detail = fftconvolve(frame, hi, mode='full')[1:][::2]
        frame = fftconvolve(frame, lo, mode='full')[1:][::2]
        ans.append(detail)
    ans.append(frame)
    ans.reverse()
    return ans

def idwt(t, lo, hi):
    '''
    Compute the inverse discrete wavelet transform of a list of 
    transform coefficients with respect to the wavelet filters
    lo and hi.
    Inputs:
        t -- a list containing the frame and detail coefficients of
             a signal, corresponding to the output of dwt.
        lo -- numpy array giving the lo-pass filter
        hi -- numpy array giving the hi-pass filter     
    Outputs:
        f -- a numpy array giving the recovered signal.
    '''
    f = t[0]
    for i in xrange(len(t)-1):
        det = t[i+1]
        frame = np.zeros(len(f)*2)
        frame[::2] = f
        frame = fftconvolve(frame, lo, mode='full')[:-1]
        detail = np.zeros(len(det)*2)
        detail[::2] = det
        detail = fftconvolve(detail, hi, mode='full')[:-1]
        f = detail + frame
    return f
