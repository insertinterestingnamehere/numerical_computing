import numpy as np
import matplotlib . pyplot as plt
from colorsys import hls_to_rgb

def colorize(z):
    zy=np.flipud(z)
    r = np.abs(zy)
    arg = np.angle(zy)

    h = (arg + np.pi)  / (2 * np.pi) + 0.5
    l = 1.0 - 1.0/(1.0 + r**0.3)
    s = 0.8

    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0,2)
    c = c.swapaxes(0,1)
    return c

def plot_complex(p, xbounds=(-1, 1), ybounds=(-1, 1), res=401):
    x = np.linspace(xbounds[0],xbounds[1], res)
    y = np.linspace(ybounds[0],ybounds[1], res)
    X,Y = np.meshgrid(x,y)
    Z=p(X+Y*1j)
    Zc=colorize(Z)
    plt.imshow(Zc,extent=(xbounds[0],xbounds[1],ybounds[0],ybounds[1]))
    plt.show()