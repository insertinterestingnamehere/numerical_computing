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
