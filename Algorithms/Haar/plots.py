import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

from matplotlib import pyplot as plt
import numpy as np
import pywt
import scipy.misc

lena = scipy.misc.lena()

def dwt2():
    lw = pywt.wavedec2(lena, 'db4', level=1)
    plt.subplot(221)
    plt.imshow(lw[0], cmap=plt.cm.Greys_r)
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(np.abs(lw[1][0]), cmap=plt.cm.Greys_r, interpolation='none') 
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(np.abs(lw[1][1]), cmap=plt.cm.Greys_r, interpolation='none')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(np.abs(lw[1][2]), cmap=plt.cm.Greys_r, interpolation='none')
    plt.axis('off')
    plt.savefig('dwt2.pdf')
    plt.clf()
    

def hardThresh(coeffs, tau):
    for i in xrange(1,len(coeffs)):
        for c in coeffs[i]:
            c[:] = pywt.thresholding.hard(c, tau)
    return coeffs
    
    
def softThresh(coeffs, tau):
    for i in xrange(1,len(coeffs)):
        for c in coeffs[i]:
            c[:] = pywt.thresholding.soft(c, tau)
    return coeffs


def denoise():
    wave = 'db4'
    sig = 20
    tau1 = 3*sig
    tau2 = 3*sig/2
    noisyLena = lena + np.random.normal(scale = sig, size=lena.shape)
    lw = pywt.wavedec2(noisyLena, wave, level=4)
    lwt1 = hardThresh(lw, tau1)
    lwt2 = softThresh(lw, tau2)
    rlena1 = pywt.waverec2(lwt1, wave)
    rlena2 = pywt.waverec2(lwt2, wave)
    plt.subplot(131)
    plt.imshow(noisyLena, cmap=plt.cm.Greys_r)
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(rlena1, cmap=plt.cm.Greys_r)
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(rlena2, cmap=plt.cm.Greys_r)
    plt.axis('off')
    
    plt.savefig('denoise.pdf')
    plt.clf()


if __name__ == "__main__":
    dwt2()
    denoise()
