import sys
import numpy as np
from matplotlib import pyplot as plt
import math
    
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
    
def fractal_ex():
    f = lambda x:x**3-x
    Df = lambda x:3*x**2-1
    roots1 = np.array([-1,1,0])
    xmin=-1.5
    xmax=1.5
    ymin=-1.5
    ymax=1.5
    plot_basins(f,Df,roots1,xmin,xmax,ymin,ymax, numpoints=700, name='fractal_ex.png')

def fractal_hw():
    xmin=-1.5
    xmax=1.5
    ymin=-1.5
    ymax=1.5
    g = lambda x:x**3-1
    Dg = lambda x:3*x**2
    roots2 = np.array([1, ((-1 + np.sqrt(3)*1j)/2), ((-1 - np.sqrt(3)*1j)/2)])
    plot_basins(g,Dg,roots2,xmin,xmax,ymin,ymax, numpoints=700, name='fractal_hw.png')    
    
def fractal_zoom():
    f = lambda x:x**3-x
    Df = lambda x:3*x**2-1
    roots1 = np.array([-1,0,1])
    xmin=.445
    xmax=.475
    ymin=-.015
    ymax=.015
    plot_basins(f,Df,roots1,xmin,xmax,ymin,ymax, numpoints=700, name='fractal_zoom.png')

if __name__=='__main__':
    fractal_zoom()
    fractal_ex()
    fractal_hw()

