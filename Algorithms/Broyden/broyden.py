#import FiniteDiff2 as fd2
import scipy.linalg as la
from numdifftools import Jacobian
import scipy as sp
import matplotlib.pyplot as plt


def Problem1():
    def f(x):
        return sp.exp(x)-2
    
    ans, errors = secant(f, 0, 1, real=sp.log(2))
    
    #calculate the exponent of convergence
    
    
    
def secant(func, x1, x2, tol=1e-7, real=None, iters=30):
    _x1 = func(x1)
    _x2 = func(x2)
    if abs(_x1) < tol:
        return x1
    elif abs(_x2) < tol:
        return x2
    
    x1, x2 = float(x1), float(x2)
    xnew = x1 - _x1 * (x1-x2)/(_x1-_x2)
    maxiters = iters
    real = sp.nan if real is None else real
    errors = []
    while abs(func(xnew)) > tol and iters >= 0:
        xnew = x1 - func(x1) * (x1 - x2)/(func(x1)-func(x2))
        x1 = x2
        x2 = xnew
        
        errors.append(real-xnew)
        if iters > 0:
            iters -= 1
        else:
            raise StopIteration("Failed to find root in {} iterations.  Stopped at {}".format(maxiters, xnew))
    
    return (xnew, errors) if real is not sp.nan else xnew

def broyden1d(func, xpts, tol=1e-7, iters=30):
    """Find the zero of a function between two points (accepted as a list)

    When f(pt2) < tol, we are close enough to a zero and stop"""

    #truncate list to two item and sort
    x = xpts[:2]
    x.sort()

    #check our original points for zeros
    if abs(func(x[0])) < tol:
        return x[0]
    elif abs(func(x[1])) < tol:
        return x[1]

    ans = sp.math.log(2)
    #calculate our second guess
    x_1, x_2 = float(x[0]), float(x[1])
    l = sp.math.log(2)
    errors = []
    for i in xrange(iter):
        x_new = x_1-func(x_1)*((x_1-x_2)/(func(x_1)-func(x_2)))
        
        if abs(func(x_new)) < tol:
            return x_new, [sp.math.log(errors[i+1])/sp.math.log(errors[i]) for i in range(len(errors)-1)]
        else:
            x_1 = x_2
            x_2 = x_new
            errors.append(abs(l-x_new))
    return StopIteration("No Zeros found in {} iterations".format(iter))
        

def regula_falsi(func, xpts, tol=0.00005, iter=30):
    """Find the zero of a function between two points (accepted as a list)

    This method uses the regula falsi secant/bisection method to converge on the
    root."""

    #truncate list to two tiems and sort
    x=xpts[:2]
    x.sort()

    #check that func(x[0])<0 and func(x[1])>0
    if not(func(x[0])<0 and func(x[1])>0):
        raise ValueError


    #check our original points for zeros
    if abs(func(x[0])) < tol:
        return x[0]
    elif abs(func(x[1])) < tol:
        return x[1]

    #calculate our second guess
    x_1, x_2= float(x[0]), float(x[1])
    x_old = x_2
    fx1 = func(x_1)
    for i in xrange(iter):
        x_new = x_1-fx1*((x_1-x_2)/(fx1-func(x_2)))

        if abs(x_new-x_old) < tol:
            return x_new, i
        else:
            x_old = x_new
            tmp = func(x_new)
            if x_1*x_2 > 0.0:
                x_1 = x_new
            else:
                x_2 = x_new
    return StopIteration("No Zeros found in {} iterations".format(iter))


def broyden(func, x1, x2, tol=1e-5, maxiter=50):
    """Calculate the zero of a multi-dimensional function using Broyden's method"""
    
    def isscalar(x):
        if isinstance(x, sp.ndarray):
            if x.size == 1:
                return x.flatten()[0]
            else:
                return x
        else:
            return x

    def update_Jacobian(preJac, ch_x, ch_F):
        """Update Jacobian from preX to newX
        preX and newX are assumed to be array objects of the same shape"""
                
        frac = (ch_F-(preJac.dot(ch_x)))/(la.norm(ch_x)**2)

        Jac = preJac+sp.dot(isscalar(frac),ch_x.T)
        return Jac
        
    #truncate list to two tiems and sort
    x1 = sp.vstack(x1.flatten())
    x2 = sp.vstack(x2.flatten())
    
    fx1 = func(x1)
    fx2 = func(x2)
    
    #check our original points for zeros
    if abs(fx1) < tol:
        return x1
    elif abs(fx2) < tol:
        return x2

    #Calculate initial Jacobian matrix
    jac = Jacobian(func)(x1)
    mi = maxiter
    while abs(fx2) > tol and mi > 0:        
        fx1 = func(x1)
        fx2 = func(x2)
        ch_x=x2-x1
        ch_F=fx2-fx1
        
        jac = update_Jacobian(jac, ch_x, ch_F)
        y = la.lstsq(jac, sp.array([-fx2]))[0]
        xnew = y+x2
        x1 = x2
        x2 = xnew
        mi -= 1
    
    if mi==0:
        raise StopIteration("Did not converge in {} iterations".format(maxiter))
    else:
        return x2, maxiter-mi
        
def broydeninv(func, x1, x2, tol=1e-5, maxiter=50):
    """Calculate the zero of a multi-dimensional function using Broyden's method"""
    
    def isscalar(x):
        if isinstance(x, sp.ndarray):
            if x.size == 1:
                return x.flatten()[0]
            else:
                return x
        else:
            return x

    def update_Jacobian(preJac, ch_x, ch_F):
        """Update Jacobian from preX to newX
        preX and newX are assumed to be array objects of the same shape"""
        
        numer = ch_x-preJac*ch_F

        denom = ch_x.T.dot(preJac).dot(ch_F)

        outside = ch_x.T.dot(preJac)

        Jac = preJac+sp.dot(numer/denom,outside)
        return Jac
        
    #truncate list to two tiems and sort
    x1 = sp.vstack(x1)
    x2 = sp.vstack(x2)
    
    fx1 = func(x1)
    fx2 = func(x2)
    
    #check our original points for zeros
    if abs(fx1) < tol:
        return x1
    elif abs(fx2) < tol:
        return x2

    #Calculate initial Jacobian matrix
    jac = Jacobian(func)(x1)
    
    jac = la.pinv(jac)
    mi = maxiter
    while abs(fx2) > tol and mi > 0:        
        fx1 = func(x1)
        fx2 = func(x2)
        ch_x=x2-x1
        ch_F=fx2-fx1
        
        jac = update_Jacobian(jac, ch_x, ch_F)
        
        y = sp.dot(jac, [-1.0*fx2])
        xnew = y+x2
        
        x1 = x2
        x2 = xnew
        mi -= 1
    
    if mi==0:
        raise StopIteration("Did not converge in {} iterations".format(maxiter))
    else:
        return x2, maxiter-mi