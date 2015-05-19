import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import matplotlib.pyplot as plt
import numpy as np
import sys

def basins_1d():
    f = lambda x:x**2-1
    Df = lambda x:2*x
    x0 = np.linspace(-1.5, 1.5, 40)
    
    xold=x0
    n = 0
    while n <= 6:
        xnew = xold - f(xold)/Df(xold)
        xold = xnew
        n += 1
    
    plt.scatter(x0, np.zeros_like(x0), marker='s', c=xnew, edgecolor='None', cmap='bwr') 
    plt.plot(x0, f(x0), c='black', linewidth=3)
    plt.savefig('basins1d.pdf', bbox_inches='tight')
    plt.close()
    
def fractal_1d():
    f = lambda x:x**3-x
    Df = lambda x:3*x**2-1
    x0 = np.linspace(-1.5, 1.5, 500)
    
    xold=x0
    n = 0
    while n <= 50:
        xnew = xold - f(xold)/Df(xold)
        xold = xnew
        n += 1
    
    y = np.array([-.1, .1])
    X, Y = np.meshgrid(x0, y)
    plt.pcolormesh(X, Y, np.atleast_2d(xnew).repeat(2, axis=0), cmap='brg')
    plt.plot(x0, f(x0), c='black', linewidth=3)
    plt.savefig('fractal1d.pdf', bbox_inches='tight')
    plt.close()   
    
def plot_basins(f, Df, roots, xmin, xmax, ymin, ymax, numpoints=100, iters=15, colormap='brg', name='name.png', dpinum=150):
    xreal = np.linspace(xmin, xmax, numpoints)
    ximag = np.linspace(ymin, ymax, numpoints)
    Xreal, Ximag = np.meshgrid(xreal, ximag)
    Xold = Xreal + 1j * Ximag
    for i in xrange(iters):
        Xnew = Xold - f(Xold)/Df(Xold)
        Xold = Xnew
    m,n = Xnew.shape
    for i in xrange(m):
        for j in xrange(n):
            Xnew[i,j] = np.argmin(np.abs(Xnew[i,j]-roots))    
    plt.pcolormesh(Xreal, Ximag, Xnew, cmap=colormap)
    plt.savefig(name, bbox_inches='tight', dpi=dpinum)

if __name__ == "__main__":
    basins_1d()
    fractal_1d()
