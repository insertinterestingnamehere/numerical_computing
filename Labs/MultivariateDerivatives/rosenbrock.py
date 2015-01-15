# Code obtained from http://matplotlib.1069221.n5.nabble.com/How-to-shift-colormap-td18451.html
#Modified and used to shift color map on Rosenbrock function

import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
matplotlib.rcParams['savefig.bbox'] = 'standard'

import math
import copy
import numpy
from matplotlib import pyplot, colors, cm
import scipy as np
from mpl_toolkits.mplot3d import Axes3D

def cmap_powerlaw_adjust(cmap, a):
    '''
    returns a new colormap based on the one given
    but adjusted via power-law:

    newcmap = oldcmap**a
    '''
    if a < 0.:
        return cmap
    cdict = copy.copy(cmap._segmentdata)
    fn = lambda x : (x[0]**a, x[1], x[2])
    for key in ('red','green','blue'):
        cdict[key] = map(fn, cdict[key])
        cdict[key].sort()
        assert (cdict[key][0]<0 or cdict[key][-1]>1), \
            "Resulting indices extend out of the [0, 1] segment."
    return colors.LinearSegmentedColormap('colormap',cdict,1024)

def cmap_center_adjust(cmap, center_ratio):
    '''
    returns a new colormap based on the one given
    but adjusted so that the old center point higher
    (>0.5) or lower (<0.5)
    '''
    if not (0. < center_ratio) & (center_ratio < 1.):
        return cmap
    a = math.log(center_ratio) / math.log(0.5)
    return cmap_powerlaw_adjust(cmap, a)

def cmap_center_point_adjust(cmap, range, center):
    '''
    converts center to a ratio between 0 and 1 of the
    range given and calls cmap_center_adjust(). returns
    a new adjusted colormap accordingly
    '''
    if not ((range[0] < center) and (center < range[1])):
        return cmap
    return cmap_center_adjust(cmap,
        abs(center - range[0]) / abs(range[1] - range[0]))


def rosenbrock():
    def f(x, y):
        return (1.0-x)**2 + 100*(y-x**2)**2

    a = np.arange(-1.8, 1.8,.01)
    b = np.arange(-1,2.5,.01)
    A, B = np.meshgrid(a,b)
    Z = f(A, B)

    plotkwargs = {'rstride': 8,
                'cstride': 8,
                'linewidth': 0.01}

    fig = pyplot.figure(figsize=(6,4))
    ax = fig.add_subplot(111, projection='3d')
    cmap = cm.jet
    plt = ax.plot_surface(A, B, Z, cmap=cmap, **plotkwargs)
    plt.set_cmap(cmap_center_adjust(cmap, .25))
    ax.view_init(elev=48, azim=-125)
    pyplot.savefig('Rosenbrock.pdf')


if __name__ == '__main__':
    rosenbrock()
    
