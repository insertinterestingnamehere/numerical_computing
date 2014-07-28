import numpy as np
import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
from matplotlib import pyplot as plt
from matplotlib.cm import jet as cmap
from mpl_toolkits.mplot3d import Axes3D
from mayavi import mlab

def singular_surface_plot(f, filename, mn=-1., mx=1., res=500, threshold=2., lip=.1, kind='real'):
    x = np.linspace(mn, mx, res)
    X, Y = np.meshgrid(x, x, copy=False)
    Z = f(X + 1.0j * Y)
    if kind == 'real':
        Z = Z.real
    elif kind == 'imag':
        Z = Z.imag
    elif kind == 'abs':
        Z = np.absolute(Z)
    else:
        raise NotImplementedError("Kind must be 'real', 'imag', or 'abs'.")
    Z[(threshold+lip>Z)&(Z>threshold)] = threshold
    Z[(-threshold-lip<Z)&(Z<-threshold)] = -threshold
    Z[np.absolute(Z) >= threshold + lip] = np.nan
    mlab.mesh(X, Y, Z)
    mlab.savefig(filename)
    mlab.clf()

def singular_color_plot(f, filename, mn=-.5, mx=.5, res=500, kind='real'):
    x = np.linspace(mn, mx, res)
    X, Y = np.meshgrid(x, x, copy=False)
    Z = f(X + 1.0j * Y)
    if kind == 'real':
        Z = Z.real
    elif kind == 'imag':
        Z = Z.imag
    elif kind != 'abs':
        raise ValueError("Kind must be 'real', 'imag' or 'abs'.")
    Z = np.sin(np.log(np.absolute(Z)))
    Z[np.isnan(Z) | np.isinf(Z)] = 0
    plt.imsave(filename, Z, cmap=cmap)

def contour_1():
    t = np.linspace(0,1,41)
    cx = np.empty(81)
    cy = np.empty(81)
    r = 2
    cx[:41] =  2 * r * t - r
    cy[:41] = 0.
    cx[40:] = r * np.cos(np.pi * t)
    cy[40:] = r * np.sin(np.pi * t)
    plt.xlim((-3,3))
    plt.ylim((-1,3))
    plt.axes()
    plt.plot(cx, cy)
    plt.savefig("contour1.pdf")
    plt.clf()

def contour_2():
    t = np.linspace(0,1,41)
    cx = np.empty(161)
    cy = np.empty(161)
    r = 2
    cx[:41] = (r - 1./r) * t + 1./r
    cy[:41] = 0.
    cx[80:121] = -cx[:41]
    cy[80:121] = 0.
    cx[40:81] = r * np.cos(np.pi * t)
    cy[40:81] = r * np.sin(np.pi * t)
    cx[120:] = np.cos(np.pi * t[::-1]) / r
    cy[120:] = np.sin(np.pi * t[::-1]) / r
    plt.xlim((-3,3))
    plt.ylim((-1,3))
    plt.axes()
    plt.plot(cx, cy)
    plt.savefig("contour2.pdf")
def colorize(z):
    zy=np.flipud(z)
    r = np.abs(zy)
    arg = np.angle(zy) 

    h = (arg + np.pi)  / (2 * np.pi) + 0.5
    l = 1.0 - 1.0/(1.0 + r**0.3)
    s = 0.8

    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c) 
    c = c.swapaxes(0,2) 
    c = c.swapaxes(0,1) 
    return c

def plot_complex(p, xbounds=(-1, 1), ybounds=(-1, 1), res=401,name="False"):
    x = np.linspace(xbounds[0],xbounds[1], res)
    y = np.linspace(ybounds[0],ybounds[1], res)
    X,Y = np.meshgrid(x,y)
    Z=p(X+Y*1j)
    #Z[np.isnan(Z)]=np.infty
    Zc=colorize(Z)
    plt.imshow(Zc,extent=(xbounds[0],xbounds[1],ybounds[0],ybounds[1]))
    if name != "False":
        plt.savefig(name)
    plt.show()

if __name__ == '__main__':
    # Surface plots of singularities.
    f = lambda z: 1 / z
    singular_surface_plot(f, 'inv_abs_surface.png', kind='abs')
    plot_complex(f,name='invz')
    f = lambda z: 1 / z**2
    singular_surface_plot(f, 'inv2_abs_surface.png', kind='abs')
    plot_complex(f,name='invz2')
    # Show a matplotlib windows so mayavi can close properly.
    #plt.plot([0, 1])
    #plt.show()
    
    # Color plots of singularities
    '''
    f = lambda z: 1 / z
    singular_color_plot(f, 'inv_real.png', kind='real')
    singular_color_plot(f, 'inv_imag.png', kind='imag')
    singular_color_plot(f, 'inv_abs.png', kind='abs')
    f = lambda z: 1 / z**4
    singular_color_plot(f, 'inv4_real.png', kind='real')
    singular_color_plot(f, 'inv4_imag.png', kind='imag')
    singular_color_plot(f, 'inv4_abs.png', kind='abs')
    f = lambda z: np.exp(1 / z)
    singular_color_plot(f, 'exp_inv_real.png', kind='real')
    singular_color_plot(f, 'exp_inv_imag.png', kind='imag')
    singular_color_plot(f, 'exp_inv_abs.png', kind='abs')
    f = lambda z: np.exp(1 / z**2)
    singular_color_plot(f, 'exp_inv2_real.png', kind='real')
    singular_color_plot(f, 'exp_inv2_imag.png', kind='imag')
    singular_color_plot(f, 'exp_inv2_abs.png', kind='abs')
    contour_1()
    contour_2()
    '''
