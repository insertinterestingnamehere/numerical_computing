import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import numpy as np
from matplotlib import pyplot as plt
import filters
import FiniteDiff

K = plt.imread('cameraman.tif')
blur = np.array([[2,4,5,4,2],
                 [4,9,12,9,4],
                 [5,12,15,12,5],
                 [4,9,12,9,4],
                 [2,4,5,4,2]])/159.
    
def cameramanClean():
    plt.imsave('cameramanClean.pdf', np.flipud(K), origin='lower')
    plt.clf()
    
def cameramanBlur():
    O = filters.Filter(K, blur)
    plt.imsave('cameramanBlur.pdf', np.flipud(O), origin='lower')
    plt.clf()

def edges():
    S = np.array([[-1, -2, -1],
                  [0, 0, 0],
                  [1, 2, 1]])/8.
    
    #filter the image horizontally and vertically to get gradient values
    Oy = filters.Filter(K, S)
    Ox = filters.Filter(K, S.T)
    
    #combine to obtain gradient magnitude at each pixel
    O = np.sqrt(Oy**2 + Ox**2)
    
    #set threshold value
    thresh = 4 * O.mean()
    
    #plot the thresholded image
    plt.imsave('edges.pdf', np.flipud(O>thresh), origin='lower')
    plt.clf()


def convergence():
    '''
    Plot the convergence of the derivative approximation.
    '''
    f = np.cos
    actual = -np.sin(3.0)
    hvals = np.linspace(1e-5, 1e-1)
    err1 = np.zeros(hvals.shape)
    err2 = np.zeros(hvals.shape)
    for i in xrange(len(hvals)):
        err1[i] = np.abs(actual - FiniteDiff.der(f, np.array([3.0]), mode='forward', h=hvals[i], o=1))
        err2[i] = np.abs(actual - FiniteDiff.der(f, np.array([3.0]), mode='forward', h=hvals[i], o=2))
    plt.subplot(121)
    plt.loglog(hvals, err1)
    plt.ylim((1e-11, 1e-1))
    plt.subplot(122)
    plt.loglog(hvals, err2)
    plt.ylim((1e-11, 1e-1))
    plt.savefig('convergence.pdf')
    plt.clf()


if __name__ == "__main__":
    cameramanClean()
    cameramanBlur()
    edges()
    convergence()
