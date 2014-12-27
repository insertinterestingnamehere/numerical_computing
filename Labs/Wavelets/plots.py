import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

from matplotlib import pyplot as plt
from scipy.misc import imread
import numpy as np
import pywt
import scipy.misc
import solution


def dwt1D_example():
    '''
    Create a plot of the discrete wavelet transform of a one-dimensional signal.
    '''
    
    end = 4*np.pi
    dom = np.linspace(0,end,1024)
    noisysin = np.sin(dom) + np.random.randn(1024)*.1
    L = np.ones(2)/np.sqrt(2)
    H = np.array([-1,1])/np.sqrt(2)
    coeffs = solution.dwt(noisysin,L,H,4)

    ax = plt.subplot(611)
    ax.plot(dom, noisysin)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel('$X$    ', rotation='horizontal')
    
    ax = plt.subplot(612)
    ax.plot(np.linspace(0,end,len(coeffs[0])), coeffs[0])
    ax.set_xticks([])
    ax.set_yticks([]) 
    ax.set_ylabel('$A_4$    ', rotation='horizontal')
    
    ax = plt.subplot(613)
    ax.plot(np.linspace(0,end,len(coeffs[1])), coeffs[1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel('$D_4$    ', rotation='horizontal')
    
    ax = plt.subplot(614)
    ax.plot(np.linspace(0,end,len(coeffs[2])), coeffs[2])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel('$D_3$    ', rotation='horizontal')
    
    ax = plt.subplot(615)
    ax.plot(np.linspace(0,end,len(coeffs[3])), coeffs[3])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel('$D_2$    ', rotation='horizontal')
    
    ax = plt.subplot(616)
    ax.plot(np.linspace(0,end,len(coeffs[4])), coeffs[4])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel('$D_1$    ', rotation='horizontal')
    plt.savefig('dwt1D.pdf')
    plt.clf()

def dwt2():
    mandrill = imread("baboon.png")
    mandrill = mandrill.mean(axis=-1)
    lw = pywt.wavedec2(mandrill, 'db4',mode='per', level=2)
    plt.imsave("mandrill1.png",lw[0], cmap=plt.cm.Greys_r)
    plt.imsave("mandrill2.png",np.abs(lw[1][0]), cmap=plt.cm.Greys_r) 
    plt.imsave("mandrill3.png",np.abs(lw[1][1]), cmap=plt.cm.Greys_r)
    plt.imsave("mandrill4.png",np.abs(lw[1][2]), cmap=plt.cm.Greys_r)
    

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
    dwt1D_example()

