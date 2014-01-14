import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.misc import derivative

# Derivative function from the numerical derivative lab.
def der(fc, x, h=.0001, degree=1, type='centered', accuracy=2):
    """ Computes the numerical of the callable function 'fc at all the
    points in array 'x'. 'degree' is the degree of the derivative to be
    computed. 'type' can be 'centered', 'forward', or 'backward'.
    'accuracy' is the desired order of accuracy. For forward and backward
    differences it can take a value of 1, 2, or 3. For centered differences
    it can take a value of 2, 4, or 6."""
    # Use these lists to manage the different coefficient options.
    A = np.array([[[0., 0., -.5, 0., .5, 0., 0.],
                   [0., 1/12., -2/3., 0., 2/3., -1/12., 0.],
                   [-1/60., 3/20., -3/4., 0., 3/4., -3/20., 1/60.]],
                  [[0., 0., 1., -2., 1., 0., 0.],
                   [0., -1/12., 4/3., -5/2., 4/3., -1/12., 0.],
                   [1/90., -3/20., 3/2., -49/18., 3/2., -3/20., 1/90.]]])
    B = np.array([[[-1., 1., 0., 0., 0.],
                   [-1.5, 2., -.5, 0., 0.],
                   [-11/6., 3., -1.5, 1/3., 0.]],
                  [[1., -2., 1., 0., 0.],
                   [2., -5., 4., -1., 0.],
                   [35/12., -26/3., 19/2., -14/3., 11/12.]]])
    if type == "centered":
        acc = int(accuracy/2) - 1
    else:
        acc = int(accuracy) - 1
    if int(degree) not in [1, 2]:
        raise ValueError ("Only first and second derivatives are supported")
    if acc not in [0, 1, 2]:
        raise ValueError ("Invalid accuracy")
    if type == 'centered':
        xdifs = np.array([fc(x+i*h) for i in xrange(-3, 4)])
        return np.inner(A[degree-1,acc], xdifs.T) / h**degree
    elif type == 'forward':
        xdifs = np.array([fc(x+i*h) for i in xrange(5)])
        return np.inner(B[degree-1,acc], xdifs.T) / h**degree
    elif type == 'backward':
        xdifs = np.array([fc(x-i*h) for i in xrange(5)])
        return np.inner(B[degree-1,acc], xdifs.T) / (-h)**degree
    else:
        raise ValueError ("invalid type")

# Partial derivative function used in the Jacobian function.
def partial(fc, x, i, h=.0001, ty="centered", ac=2):
    """ Computes a partial derivative with respect to index 'i'.
    The rest of the options are the same as the numerical derivative function."""
    def fcpart(y):
        add = np.zeros(x.shape[0])
        add[i] = y
        return fc(x+add)
    return der(fcpart, 0., h=h, type=ty, accuracy=ac)

# Numerical Jacobian function from the MultiDeriv lab.
def jac(fc, x, ty="centered", ac=2, h=.0001):
    """Compute the Jacobian matrix of a function.
    'fc' is a callable function that operates on a 1D array.
    'x' is where to evaluate the Jacobian matrix.
    Dimensions of the domain and range are infered from 'x'
    and the output of 'fc'."""
    return np.array([partial(fc, x, [i], h=h, ty=ty, ac=ac) for i in xrange(x.size)]).T

# Problem 1.
def newton(G, f, f1=None, maxiters=100, tol=1E-8, h=1E-7):
    """ Perform Newton's method for function 'f' at the points
    in the array 'G'.
    'f1' is an optional derivative function.
    'maxiters' is the maximum number of iterations.
    'tol' is the tolerance used as a stopping criterion.
    'h' is the difference used for the numerical derivatives."""
    A = np.array(G, order='C')
    C = np.zeros_like(G, dtype=bool, order='C')
    convergence = False
    if f1 is not None:
        fder = f1
    else:
        fder = lambda x: (f(x+h) - f(x)) / h
    for index, value in np.ndenumerate(A):
        if maxiters > 0:
            previous = value
            value -= f(value) / fder(value)
            if abs(value - previous) < tol:
                convergence = True
            for i in xrange(maxiters-1):
                previous = value
                value -= f(value) / fder(value)
                if abs(previous - value) < tol:
                    C[index] = True
                    break
            A[index] = value
    return A, C

# Problem 3
def multinewton(v, f, jacobian=None, maxiters=5, tol=1E-5, h=1E-7):
    """ Performs Newton's method in multiple dimensions.
    'v' is the starting vector.
    'f' is the function that accepts 'v' as an argument.
    'jacobian' is an optional function that computes the Jacobian matrix.
    'maxiters' is the maximum number of iterations.
    'tol' is the tolerance used as a stopping criterion.
    'h' is the difference used for the numerical derivatives."""
    arr = v.copy()
    prev = np.empty_like(v)
    convergence = False
    if jacobian is not None:
        j = jacobian
    else:
        j = lambda v: jac(f, v, h=h)
    for i in xrange(maxiters):
        prev[:] = arr
        arr -= la.solve(j(arr), f(arr))
        prev -= arr
        prev *= prev
        print f(arr)
        if prev.max() < tol:
            convergence=True
            break
    return arr, convergence

# Problem 4
def polyjulia(p, xmin, xmax, ymin, ymax, res=401, iters=100, tol=1E-12):
    """ Plot the Julia set of a polynomial.
    Use a 'res'x'res' grid of complex numbers with real part
    ranging from 'xmin' to 'xmax' and imaginary part
    ranging from 'ymin' to 'ymax'.
    'p' is assumed to be a numpy poly1d object, or
    at least some callable object with a 'deriv' method that
    returns its derivative and a 'roots' attribute that
    contains an array with the values of all the functions roots.
    'iters' is the number of iterations to perform.
    'tol' is the tolerance used to distinguish between
    the roots of the polynomial."""
    x = np.linspace(xmin, xmax, res)
    y = np.linspace(ymin, ymax, res)
    X, Y = np.meshgrid(x, y, copy=False)
    Z = X + 1.0j * Y
    p2 = p.deriv()
    for i in xrange(500):
        Z -= p(Z) / p2(Z)
    for index, root in np.ndenumerate(p.roots):
        Z[np.absolute(Z-root)<tol] = index
    Z[np.isnan(Z)] = p.roots.size
    plt.pcolormesh(X, Y, np.absolute(Z), cmap=plt.get_cmap('winter'))
    plt.show()

# Examples from problem 4.
def polyplot():
    """ Plot the examples in the lab."""
    for coefs, xmin, xmax, ymin, ymax in [
            ([1, -2, -2, 2], -.5, 0, -.25, .25),
            ([3, -2, -2, 2], -1, 1, -1, 1),
            ([1, 3, -2, -2, 2], -1, 1, -1, 1),
            ([1, 0, 0, -1], -1, 1, -1, 1)]:
        polyjulia(np.poly1d(coefs), xmin, xmax, ymin, ymax)

# Problem 5.
def mandelbrot(xmin=-1.5, xmax=.5, ymin=-1, ymax=1, guess=complex(0,0), res=401, iters=200):
    """ Plot the Mandelbrot set."""
    x = np.linspace(xmin, xmax, res)
    y = np.linspace(ymin, ymax, res)
    X, Y = np.meshgrid(x, y, copy=False)
    Z = X + 1.0j * Y
    vals = np.empty_like(Z)
    vals[:] = guess
    for i in xrange(iters):
        vals[:] = vals**2 + Z
    vals[np.isnan(vals)] = 1
    vals[np.absolute(vals)>1] = 1
    vals[np.absolute(vals)<1] = 0
    plt.pcolormesh(X, Y, np.absolute(vals), cmap=plt.get_cmap('winter'))
    plt.show()

# Show the plots if the script is run.
if __name__=='__main__':
    polyplot()
    mandelbrot()
