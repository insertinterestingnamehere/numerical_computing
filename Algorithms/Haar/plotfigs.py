import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

from matplotlib import pyplot as plt
import solution
import scipy as sp
import numpy as np
import pywt

# Generate the sine curve
def sineCurve():
    pts = sp.linspace(0,2*sp.pi,256)
    plt.plot(pts,sp.sin(pts))
    plt.savefig('sinecurve.pdf')
    plt.clf()
    
# Generate the discretized sine curve
def discSineCurve():
    frame_4 = solution.getFrame(2)
    plt.plot(sp.linspace(0,2*sp.pi,len(frame_4)),frame_4,drawstyle='steps')
    plt.savefig('discreteSineCurve.pdf')
    plt.clf()

# Generate the detail for the sine curve
def sineCurveDetail():    
    detail = solution.getDetail(2)
    detail[-1] = detail[-2]
    b = []
    for i in detail:
        b.extend([i,-i])
    plt.plot(sp.linspace(0,2*sp.pi,len(b)),b,drawstyle='steps')
    plt.savefig('sineCurveDetail.pdf')
    plt.clf()

# Generate the Mexican Hat Wavelet image
def mexicanHat():
    def mex(sigma,t):
        return (2.0/sp.sqrt(3*sigma*sp.sqrt(sp.pi)))*(1-(1.0*t**2/sigma**2))*sp.exp(-t**2*1.0/(2*sigma**2))
    x = sp.linspace(-10,10,500)
    plt.plot(x,mex(2,x))
    plt.savefig('mexicanHat.pdf')
    plt.clf()

def dwt1D():
    '''
    Create a plot of the discrete wavelet transform of a one-dimensional signal.
    '''
    db3 = pywt.Wavelet('db3')
    dom = np.linspace(1,6,2048)
    noisysin = (5-6*np.exp(-dom))*np.sin(np.exp(-dom+5)) + np.random.normal(scale=.2,size=2048)
    coeffs = pywt.wavedec(noisysin, db3, level=4)
    
    ax = plt.subplot(611)
    ax.plot(dom, noisysin)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel('f', rotation='horizontal')
    
    ax = plt.subplot(612)
    ax.plot(np.linspace(1,6,len(coeffs[0])), coeffs[0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel('a4', rotation='horizontal')
    
    ax = plt.subplot(613)
    ax.plot(np.linspace(1,6,len(coeffs[1])), coeffs[1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel('d4', rotation='horizontal')
    
    ax = plt.subplot(614)
    ax.plot(np.linspace(1,6,len(coeffs[2])), coeffs[2])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel('d3', rotation='horizontal')
    
    ax = plt.subplot(615)
    ax.plot(np.linspace(1,6,len(coeffs[3])), coeffs[3])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel('d2', rotation='horizontal')
    
    ax = plt.subplot(616)
    ax.plot(np.linspace(1,6,len(coeffs[4])), coeffs[4])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel('d1', rotation='horizontal')
    plt.savefig('dwt1D.pdf')
    plt.clf()
#dwt1D()

def dwt2D():
    '''
    Create a plot of a 2D discrete wavelet transform of the Lena image.
    '''
    l = sp.misc.lena()
    coeffs = pywt.wavedec2(l, pywt.Wavelet('haar'), level=2)
    m = l.max()
    D = np.hstack((coeffs[0]*m/coeffs[0].max(), coeffs[1][0]*m/coeffs[1][0].max()))
    D = np.vstack((D, np.hstack((coeffs[1][1]*m/coeffs[1][1].max(), coeffs[1][2]*m/coeffs[1][2].max()))))
    ax = plt.subplot(111)
    ax.imshow(np.abs(D), cmap = plt.cm.Greys_r, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig('dwt2D.pdf')
    plt.clf()
dwt2D()

#sineCurve()
#discSineCurve()
#sineCurveDetail()
#mexicanHat()   
    
