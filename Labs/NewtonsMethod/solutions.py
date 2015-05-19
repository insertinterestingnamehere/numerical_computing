import numpy as np
from matplotlib import pyplot as plt

def Newtons_method(f, x0, Df, iters=15, tol=.002):
    '''Use Newton's method to approximate a zero of a function.
    
    INPUTS:
    f     - A function handle. Should represent a function from 
            R to R.
    x0    - Initial guess. Should be a float.
    Df    - A function handle. Should represent the derivative 
            of `f`.
    iters - Maximum number of iterations before the function 
            returns. Defaults to 15.
    tol   - The function returns when the difference between 
            successive approximations is less than `tol`.
    
    RETURN:
    A tuple (x, converged, numiters) with
    x           - the approximation to a zero of `f`
    converged   - a Boolean telling whether Newton's method 
                converged
    numiters    - the number of iterations the method computed
    '''
    xold = float(x0)
    n = 0
    converged = False
    while n <= iters:
        xnew = xold - f(xold)/Df(xold)
        if np.abs(xnew-xold) < tol:
            converged = True
            break
        else:
            xold = xnew
            n += 1
            print xold
    return (xnew, converged, n)
        
    
def plot_basins(f, Df, roots, xmin, xmax, ymin, ymax, numpoints=100, iters=15, colormap='brg'):
    '''Plot the basins of attraction of f.
    
    INPUTS:
    f       - A function handle. Should represent a function 
            from C to C.
    Df      - A function handle. Should be the derivative of f.
    roots   - An array of the zeros of f.
    xmin, xmax, ymin, ymax - Scalars that define the domain 
            for the plot.
    numpoints - A scalar that determines the resolution of 
            the plot. Defaults to 100.
    iters   - Number of times to iterate Newton's method. 
            Defaults to 15.
    colormap - A colormap to use in the plot. Defaults to 'brg'.    
    '''
    xreal = np.linspace(xmin, xmax, numpoints)
    ximag = np.linspace(ymin, ymax, numpoints)
    Xreal, Ximag = np.meshgrid(xreal, ximag)
    xold = Xreal+1j*Ximag
    n = 0
    while n <= iters:
        xnew = xold - f(xold)/Df(xold)
        xold = xnew
        n += 1 

    converged_to = np.empty_like(xnew)
    for i in xrange(xnew.shape[0]):
        for j in xrange(xnew.shape[1]):
            root = np.abs(roots-xnew[i,j]).argmin()
            converged_to[i,j] = root

    plt.pcolormesh(Xreal, Ximag, converged_to, cmap=colormap)
