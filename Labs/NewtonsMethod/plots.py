import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import matplotlib.pyplot as plt
import numpy as np

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
    
def fractal_ex():
    f = lambda x:x**3-x
    Df = lambda x:3*x**2-1
    roots = np.array([-1, 0, 1])
    xreal = np.linspace(-1.5, 1.5, 700)
    ximag = np.linspace(-1.5, 1.5, 700)
    Xreal, Ximag = np.meshgrid(xreal, ximag)
    xold = Xreal+1j*Ximag
    n = 0
    while n <= 15:
        xnew = xold - f(xold)/Df(xold)
        xold = xnew
        n += 1 

    converged_to = np.empty_like(xnew)
    for i in xrange(xnew.shape[0]):
        for j in xrange(xnew.shape[1]):
            root = np.abs(roots-xnew[i,j]).argmin()
            converged_to[i,j] = root

    plt.pcolormesh(Xreal, Ximag, converged_to, cmap='brg')
    #plt.colorbar()
    plt.savefig('fractal_ex.png', bbox_inches='tight')
    plt.close()
    
def fractal_zoom():
    f = lambda x:x**3-x
    Df = lambda x:3*x**2-1
    roots = np.array([-1, 0, 1])
    xreal = np.linspace(.445, .475, 700)
    ximag = np.linspace(-.015, .015, 700)
    Xreal, Ximag = np.meshgrid(xreal, ximag)
    xold = Xreal+1j*Ximag
    n = 0
    while n <= 15:
        xnew = xold - f(xold)/Df(xold)
        xold = xnew
        n += 1 

    converged_to = np.empty_like(xnew)
    for i in xrange(xnew.shape[0]):
        for j in xrange(xnew.shape[1]):
            root = np.abs(roots-xnew[i,j]).argmin()
            converged_to[i,j] = root

    plt.pcolormesh(Xreal, Ximag, converged_to, cmap='brg')
    #plt.colorbar()
    plt.savefig('fractal_zoom.png', bbox_inches='tight')
    plt.close()
    
def fractal_hw():
    f = lambda x:x**3-1
    Df = lambda x: 3*x**2
    roots = np.array([1, -.5+np.sqrt(3)/2*1j, -.5-np.sqrt(3)/2*1j]) 
    xreal = np.linspace(-1.5, 1.5, 1000)
    ximag = np.linspace(-1.5, 1.5, 1000)
    Xreal, Ximag = np.meshgrid(xreal, ximag)
    xold = Xreal+1j*Ximag
    n = 0
    while n <= 15:
        xnew = xold - f(xold)/Df(xold)
        xold = xnew
        n += 1 

    converged_to = np.empty_like(xnew)
    for i in xrange(xnew.shape[0]):
        for j in xrange(xnew.shape[1]):
            root = np.abs(roots-xnew[i,j]).argmin()
            converged_to[i,j] = root

    plt.pcolormesh(Xreal, Ximag, converged_to, cmap='brg')
    plt.savefig('fractal_hw.png', bbox_inches='tight')
    plt.close()
    
if __name__ == "__main__":
    basins_1d()
    fractal_1d()
    fractal_ex()
    fractal_zoom()
    fractal_hw()