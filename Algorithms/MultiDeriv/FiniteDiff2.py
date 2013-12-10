import scipy as sp
import FiniteDiff as FD

def Jacobian(func, inputs, step_size=1e-7, fdtype='c'):
    """Calculate the Jacobian of a function.

    INPUTS:
        func:       Function handle to use for Jacobian
        inputs:     Input values to function
        step_size:  Step size
        fdtype:     finite difference type ('c'entered, 'f'orward, 'b'ackward)

    RETURNS:
        jacobian:   An mxn array as specified by the tuple dim.
    """

    #test our return
    #try to find dimensions by being smart
    try:
        ndim = len(func(*inputs)), len(inputs)
    except TypeError:
        ndim = 1, len(inputs)

    jacobian = sp.zeros(ndim)

    if fdtype is 'c':
        for j in range(ndim[1]):
            jacobian[:,j] = FD.cdiff(func, inputs, vary=[j], accur=6, degree=1, tol=step_size)
    elif fdtype in ['f', 'b']:
        for j in range(ndim[1]):
            jacobian[:,j] = FD.fbdiff(func, inputs, vary=[j], accur=3, degree=1, direction=fdtype, tol=step_size)

    return jacobian

def eval_hessian(func, inputs, varyi, varyj, tol):
    #inner_pval = [x+h*tol if index in vary else x for x,index in zip(xvals, range(len(xvals)))]

    inputs = sp.asarray(inputs)
    varyi = sp.asarray(varyi)
    varyj = sp.asarray(varyj)

    f1 = func(*(inputs+(varyi+varyj)*tol))
    f2 = func(*(inputs+(varyi-varyj)*tol))
    f3 = func(*(inputs+(varyj-varyi)*tol))
    f4 = func(*(inputs-(varyi+varyj)*tol))

    test = (f1-f2-f3+f4)/(4.0*tol**2)
    return test

def Hessian(func, inputs, tol=1e-5):
    ndim = [len(inputs)]*2

    hessian = sp.zeros(ndim)

    #cache our elementary vectors
    elists = []
    for j in range(ndim[0]):
        vj=[0]*(ndim[0]-1)
        vj.insert(j,1)
        elists.append(vj)

    for i in range(ndim[0]):
        for j in range(ndim[0]):
            hessian[i,j] = eval_hessian(func, inputs, elists[i], elists[j], tol)

    return hessian

'''
The following are updated solutions corresponding to the latest version of the lab.
'''

import numpy as np
from matplotlib import pyplot as plt
import math

def Jacobian(f, m, n, pt, mode='centered', o=2, h=1e-5):
    '''
    Approximate the Jacobian of a function at a point.
    Inputs:
        f -- a callable function whose jacobian we will approximate
        m -- integer giving the dimension of the domain or f
        n -- integer giving the dimension of the range of f
        pt -- numpy array specifying the point at which to approximate the Jacobian
        mode -- specifies the type of difference scheme. Should take values in
                ['centered', 'backward', 'forward'].
        d -- the order of the derivative. Should take values in [1,2]
        o -- order of approximation. If mode = 'centered', should take values
             [2,4,6], otherwise should take values in [1,2,3]
        h -- the size of the difference step
    Returns:
        jac -- array the same shape as pts, giving the approximate Jacobian at 
               each point in pts.
    '''
    jac = np.empty((m,n))
    for i in xrange(n):
        off = np.zeros(n)
        off[i] = h
        if mode == 'centered':
            if o == 2:
                jac[:, i] = (-.5*f(pt - off) + .5*f(pt + off))/h
            if o == 4:
                jac[:, i] = (f(pt-2*off)/12 - 2*f(pt-off)/3 + 2*f(pt+off)/3 - f(pt+2*off)/12)/h
            if o == 6:
                jac[:, i] = (-f(pt-3*off)/60 + 3*f(pt-2*off)/20 - 3*f(pt-off)/4 + 
                          f(pt+3*off)/60 - 3*f(pt+2*off)/20 + 3*f(pt+off)/4)/h
        if mode == 'forward':
            if o == 1:
                jac[:, i] = (-f(pt)+f(pt+off))/h
            if o == 2:
                jac[:, i] = (-3*f(pt)/2 + 2*f(pt+off) - f(pt+2*off)/2)/h
            if o == 3:
                jac[:, i] = (-11*f(pt)/6 + 3*f(pt+off) - 3*f(pt+2*off)/2 + f(pt+3*off)/3)/h
        if mode == 'backward':
            if o == 1:
                jac[:, i] = -(-f(pt)+f(pt-off))/h
            if o == 2:
                jac[:, i] = -(-3*f(pt)/2 + 2*f(pt-off) - f(pt-2*off)/2)/h
            if o == 3:
                jac[:, i] = -(-11*f(pt)/6 + 3*f(pt-off) - 3*f(pt-2*off)/2 + f(pt-3*off)/3)/h
    return jac

def Hessian(f, n, pt, h=1e-5):
    '''
    Approximate the Hessian of a function at a given point.
    Inputs:
        f -- callable function object
        n -- integer giving the dimension of the domain of f
        pt -- flat numpy array giving the point at which to approximate
        h -- step size for difference scheme
    Returns:
        hess -- numpy array giving the approximated Hessian matrix
    '''
    hess = np.empty((n,n))
    off = np.eye(n)*h
    for i in xrange(n):
        for j in xrange(n):
            hess[i,j] = (f(pt+off[i]+off[j])-f(pt + off[i]-off[j]) -
                         f(x+off[j]-off[i]) + f(pt-off[i]-off[j]))/(4*h**2)
    return hess
