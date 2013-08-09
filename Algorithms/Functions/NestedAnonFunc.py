import scipy as sp
from scipy.optimize import bisect, fmin
from scipy.integrate import odeint

def randPoly(n):
    poly = sp.poly1d(sp.rand(n+1,))
    return poly

def findRoot(poly):
    """Find a root of a given polynomial.  If no root is found, return a local min"""

    #bisect requires a negative value and positive value to find root
    if poly.order%2==0:
        #return a local min
        print "Minimum at x=%.5f" % fmin(poly, 0)
    else:
        #find a negative value of the function
        #X = sp.linspace(-10,10, 1000)
        print "Zero at x=%.5f" % bisect(poly, poly(-10), poly(10))

def lambert_est(x):
    lambert=lambda y: y*sp.exp(y)-x
    return bisect(lambert, -100, 100)

def diffeq():
    a = lambda x, t: [x[1], -3*x[0]]

    return odeint(a, [1,0], sp.linspace(-1,1,50))

def newtonMethod(f, df, x0, tol=0.001):
    """Calculate the root of f within tolerance tol"""

    xi = x0
    while abs(f(xi)) > tol:
        xi = xi - (float(f(xi))/df(xi))
    return xi
        