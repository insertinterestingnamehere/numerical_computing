import numpy as np
import scipy as sp
from scipy.signal import fftconvolve
from scipy.misc import imread
from matplotlib import pyplot as plt
from matplotlib import cm
import pywt

############################################################################
############################# PROBLEM 1 ####################################
############################################################################

# calculate one level of the transform, LL, LH, HL, HH.
# this is probably not the most efficient implementation, but it works.
# due to reasons not completely clear to me, convolution returns an array that 
# has one too many entries.
# if you keep the odd-indexed elements, then you can go to the end.
# parameters: array of size 2^n by 2^n image, 1d array lo_d, 1d array hi_d
# lo_d and hi_d are the low-pass and hi-pass filters of the wavelet
# returns a list of 4 coefficient arrays: [LL,LH,HL,HH]
def dwt2_pass(image,lo_d,hi_d):
	temp = sp.zeros([image.shape[0], image.shape[1]/2])

    # Calculate LL and LH
	LL = sp.zeros([image.shape[0]/2, image.shape[1]/2])
    LH = sp.zeros([image.shape[0]/2, image.shape[1]/2])
	for i in xrange(image.shape[0]):
		temp[i] = sp.signal.fftconvolve(image[i], lo_d, mode='full')[1::2]
	for i in xrange(image.shape[1]/2):
		LL[:,i] = sp.signal.fftconvolve(temp[:,i],lo_d,mode='full')[1::2]
        LH[:,i] = sp.signal.fftconvolve(temp[:,i],hi_d,mode='full')[1::2]
    
    # Calculate HL and HH
	HL = sp.zeros([image.shape[0]/2, image.shape[1]/2])
    HH = sp.zeros([image.shape[0]/2, image.shape[1]/2])
	for i in xrange(image.shape[0]):
		temp[i] = sp.signal.fftconvolve(image[i], hi_d, mode='full')[1::2]
	for i in xrange(image.shape[1]/2):
		HL[:,i] = sp.signal.fftconvolve(temp[:,i],lo_d,mode='full')[1::2]
        HH[:,i] = sp.signal.fftconvolve(temp[:,i],hi_d,mode='full')[1::2]
        
	return [LL,LH,HL,HH]

# to visualize one level of coefficients:
#plt.imshow(np.vstack([np.hstack([LL,LH]), np.hstack([HL,HH])]),cmap=cm.Greys_r)
#plt.imshow()

# to visualize two levels of coefficients:
LL1,LH1,HL1,HH1 = dwt2_pass(image,lo_d,hi_d)
LL2,LH2,HL2,HH2 = dwt2_pass(LL1,lo_d,hi_d)
NW = np.vstack([np.hstack([LL2,LH2]),np.hstack([HL2,HH2])])
#plt.imshow(np.vstack([np.hstack([NW,LH1]), np.hstack([HL1,HH1])]),cmap=cm.Greys_r)
#plt.show()

# now let's try the whole decomposition:
def dwt2(image, lo_d, hi_d, level=0):
	max_level = np.floor(np.log2(image.shape[0]*1.0/len(lo_d))) + 1
    if (level < 1 or level > max_level):
        level = int(max_level)
	result = []
	sig = image
    for i in xrange(level):
		coeffs = dwt2_pass(sig,lo_d,hi_d)
		result.append(coeffs[1:])
		sig = coeffs[0]
	result.append(sig)
	result.reverse()
	return result

# Single level reconstruction
# parameters: coeffs = [LL,LH,HL,HH], lo_r and hi_r the filters
# returns: LL on the next level
def idwt2_pass(coeffs, lo_r, hi_r):
	LL, LH, HL, HH = coeffs
    n = LL.shape[0]
	temp1 = sp.zeros([2*n,n])
	temp2 = sp.zeros([2*n,n])
	up1 = sp.zeros(2*n)
	up2 = sp.zeros(2*n) 
	for i in xrange(n):
		up1[1::2] = HH[:,i]
		up2[1::2] = HL[:,i]
		temp1[:,i] = fftconvolve(up1, hi_r)[1:] + fftconvolve(up2, lo_r)[1:]
		up1[1::2] = LH[:,i]
		up2[1::2] = LL[:,i]		
		temp2[:,i] = fftconvolve(up1, hi_r)[1:] + fftconvolve(up2, lo_r)[1:]
	result = sp.zeros([2*n,2*n])
	for i in xrange(2*n):
		up1[1::2] = temp1[i]
		up2[1::2] = temp2[i]
		result[i] = fftconvolve(up1, hi_r)[1:] + fftconvolve(up2, lo_r)[1:]
	return result

# now the whole reconstruction algorithm
def idwt2(coeffs,lo_r,hi_r):
	result = coeffs[0]
	for i in xrange(len(coeffs)-1):
		args = [result]
		args.extend(coeffs[i+1])
		result = idwt2_pass(args,lo_r,hi_r)
	return result

############################################################################
############################# PROBLEM 2 ####################################
############################################################################

def plot(image):
    plt.imshow(image,cmap=cm.Greys_r)
    plt.show()

lenna = np. array ( imread (" Lenna .png",flatten = True ),dtype =np. float32 )

wave = pywt.Wavelet('haar')
lo_d = sp.array(wave.dec_lo)
hi_d = sp.array(wave.dec_hi)
lo_r = sp.array(wave.dec_lo)
hi_r = sp.array(wave.dec_hi)

coeffs = dwt2(lenna, lo_d, hi_d, 1)
coeffs[0] *= 0 
edges = idwt2(coeffs,lo_r,hi_r)
plot(np.absolute(edges))
plot(np.hstack([lenna,lenna+edges]))

############################################################################
############################# PROBLEM 3 ####################################
############################################################################

def hardThreshold(coeffs,thresh):
	new_coeffs = []
	for j in coeffs:
		new_coeffs.append(sp.copy(j))
	for j in xrange(1,len(new_coeffs)):
		for i in new_coeffs[j]:
			i *= sp.absolute(i) > thresh
	return new_coeffs

def softThreshold(coeffs,thresh):
	new_coeffs = []
	for j in coeffs:
		new_coeffs.append(sp.copy(j))
	for j in xrange(1,len(new_coeffs)):
		for i in new_coeffs[j]:
			i[sp.absolute(i)<thresh] = 0
			i[sp.absolute(i)>=thresh] -= (sp.sign(i[sp.absolute(i)>=thresh]))*thresh
	return new_coeffs

############################################################################
############################# PROBLEM 4 ####################################
############################################################################

def addGuassianNoise(image,deviation):
	return image + sp.random.normal(0.0,deviation,image.shape)

noisy = addGaussianNoise(lenna,20)
coeffs = dwt2(noisy,lo_d,hi_d,4)
denoised1 = idwt2(hardThreshold(coeffs,60),lo_r,hi_r)
denoised2 = idwt2(softThreshold(coeffs,30),lo_r,hi_r)
plot(np.hstack([noisy,denoised1,desnoised2]))

############################################################################
############################# PROBLEM 5 ####################################
############################################################################

def quantize(coeffs,step,t=2):
	new_coeffs = []
	for j in coeffs:
		new_coeffs.append(sp.copy(j))
	neg_indices = new_coeffs[0]<0
	pos_indices = np.logical_not(neg_indices)
	new_coeffs[0][neg_indices] = np.floor(new_coeffs[0][neg_indices]/step + 0.5*t)
	new_coeffs[0][pos_indices] = np.ceil(new_coeffs[0][pos_indices]/step - 0.5*t)
	for i in xrange(1,len(new_coeffs)-1):
		for j in new_coeffs[i]:
			neg_indices = j<0
			pos_indices = np.logical_not(neg_indices)
			j[neg_indices] = np.floor(j[neg_indices]/step + 0.5*t)
			j[pos_indices] = np.floor(j[pos_indices]/step - 0.5*t)
	return new_coeffs

def dequantize(coeffs,step,t=2):
	new_coeffs = []
	for j in coeffs:
		new_coeffs.append(sp.copy(j))
	neg_indices = new_coeffs[0]<0
	pos_indices = new_coeffs[0]>0
	new_coeffs[0][neg_indices] = (new_coeffs[0][neg_indices] + 0.5 - 0.5*t)*step
	new_coeffs[0][pos_indices] = (new_coeffs[0][pos_indices] - 0.5 + 0.5*t)*step
	for i in xrange(1,len(new_coeffs)-1):
		for j in new_coeffs[i]:
			neg_indices = j<0
			pos_indices = j>0
			j[neg_indices] = (j[neg_indices]+ 0.5 - 0.5*t)*step
			j[pos_indices] = (j[pos_indices]- 0.5 + 0.5*t)*step
	return new_coeffs

coeffs = dwt2(lenna,lo_d,hi_d)
step = 1
compressed = dequantize(quantize(coeffs,step),step)
plot(np.hstack([lenna,compressed]))
