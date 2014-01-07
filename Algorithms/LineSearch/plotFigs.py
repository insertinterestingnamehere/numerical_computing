import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
from matplotlib import pyplot as plt

import numpy as np
from scipy import linalg as la

def newtonsMethod1d(f, df, ddf, x, niter=10):
    '''
    Perform Newton's method to minimize a function from R to R.
    Inputs:
        f -- objective function (twice differentiable)
        df -- first derivative
        ddf -- second derivative
        x -- initial guess
        niter -- integer, giving the number of iterations
    Returns:
        the approximated minimizer
    '''
    for i in xrange(niter):
        x = x-df(x)/ddf(x)
    return x, f(x)

def backtracking(f, slope, x, p, a=1, rho=.9, c=10e-4):
    '''
    Perform a backtracking line search to satisfy the Wolfe Conditions.
    Return the step length.
    Inputs:
        f -- the objective function
        slope -- equal to grad(f)^T p
        x -- current iterate
        p -- current direction
        a -- intial step length (set to 1 in Newton and quasi-Newton methods)
        rho -- number in (0,1)
        c -- number in (0,1)
    Returns:
        the computed step size
    '''
    b = f(x)
    while f(x+a*p) > b + c*a*slope:
        a = rho*a
    return a

def gradientDescent(f, Df, x, niter=10):
    '''
    Minimize a function using gradient descent.
    Inputs:
        f -- differentiable real-valued function
        Df -- the gradient of the function
        x -- initial point
        niter -- integer giving the number of iterations to run.
    Returns:
        a list, the sequence of points generated
    '''
    pts = []
    pts.append(x.copy())
    for i in xrange(niter):
        p = -Df(x)
        slope = (p**2).sum()
        a = .2/np.sqrt(slope)
        x += a*p
        pts.append(x.copy())
    return pts

def newtonsMethod(f, Df, DDf, x, niter=10):
    '''
    Minimize a function using Newton's method.
    Inputs:
        f -- real-valued, twice-differentiable function
        Df -- the gradient of the function
        DDf -- the Hessian of the function
        x -- initial point
        niter -- integer giving the number of iterations
    Returns:
        a list, the sequence of points generated
    '''
    pts = [x.copy()]
    for i in xrange(niter):
        p = la.solve(DDf(x),-Df(x))
        slope = (p**2).sum()
        a = backtracking(f, slope, x, p)
        x += a*p
        pts.append(x.copy())
    return pts
    
    
def myFunc(x):
    return 4*x**2 - 13*x + 40 + 6*np.sin(4*x)
def myDFunc(x):
    return 8*x - 13+24*np.cos(4*x)
def myDDFunc(x):
    return 8-96*np.sin(4*x)
def f(x):
    w = np.array([1,4])
    return np.exp((x**2*w).sum())
def Df(x):
    w = np.array([1,4])
    return 2*np.exp((x**2*w).sum())*x*w
def DDf(x):
    w = np.array([1,4])
    A = np.diag(np.exp((x**2*w).sum())*(2*w+4*(x*w)**2))
    A[0,1] = 4*((x*w).prod())*np.exp((x**2*w).sum())
    A[1,0] = A[0,1]
    return A
def newton():
    x1,f1 = newtonsMethod1d(myFunc, myDFunc, myDDFunc, 1, niter=200)
    x2,f2 = newtonsMethod1d(myFunc, myDFunc, myDDFunc, 4, niter=200)
    dom = np.linspace(-10,10,100)
    plt.plot(dom, myFunc(dom))
    plt.plot(x1, f1, '*')
    plt.plot(x2, f2, '*')
    plt.annotate('Global Minimum', xy=(x1, f1), xytext=(-4, 200),
                arrowprops=dict(facecolor='black', shrink=0.1),)
    plt.annotate('Local Minimum', xy=(x2,f2), xytext=(2, 175),
                    arrowprops=dict(facecolor='black', shrink=0.1),)
    plt.savefig('newton.pdf')
    plt.clf()

def comparison():
    pts1 = np.array(newtonsMethod(f, Df, DDf, np.array([2.,1.]), niter=10))
    pts2 = np.array(gradientDescent(f, Df, np.array([2.,1.]), niter=10))
    w = np.array([1,4])
    dom = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(dom, dom)
    Z = np.exp(w[0]*X**2+w[1]*Y**2)
    vals = np.exp((pts2**2*w).sum(axis=1))
    plt.contour(X,Y,Z, vals)
    plt.plot(pts1[:,0], pts1[:,1], '*-')
    plt.plot(pts2[:,0], pts2[:,1], '*-')
    plt.savefig('comparison.pdf')
    plt.clf()


newton()
comparison()
