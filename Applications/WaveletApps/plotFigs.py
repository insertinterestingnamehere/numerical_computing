import matplotlib
#matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

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
dwt2()
