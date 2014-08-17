import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
from matplotlib import pyplot as plt

import numpy as np
from scipy.optimize import fmin_cg


def steepestDescent(Q, b, x, niter=10):
    '''
    Minimize .5x^TQx - b^Tx + c using the method of Steepest Descent.
    Inputs:
        Q -- square positive definit symmetric array (n by n).
        b -- length n array
        x -- length n array, the inital guess
        niter -- integer giving the number of iterations to perform
    Returns:
        a list of the points found in the algorithm.
    '''
    pts = []
    for i in xrange(niter):
        pts.append(x.copy())
        r = Q.dot(x) - b
        x += -((r*r).sum()/((r*(Q.dot(r))).sum()))*r
    return pts


def conjugateGradient(Q, b, x, allpts=True):
    '''
    Minimize .5x^TQx - b^Tx + c using the method of Steepest Descent.
    Equivalently, solve Qx = b.
    Inputs:
        Q -- square positive definit symmetric array (n by n).
        b -- length n array
        x -- length n array, the inital guess
    Returns:
        a numpy array, the solution to Qx = b.
    '''
    pts = []
    r = Q.dot(x) - b
    d = -r.copy()
    while not np.allclose((r*r).sum(), 0):
        if allpts:
            pts.append(x.copy())
        den = (d*Q.dot(d)).sum()
        a = -(r*d).sum()/den
        x += a*d
        r = Q.dot(x) - b
        beta = (r*Q.dot(d)).sum()/den
        d = -r+beta*d
    if allpts:
        pts.append(x)
        return pts
    else:
        return x


def steepVsConj():
    Q = np.array([[1.,0], [0,10.]])
    b = np.array([0.,0.])
    def f(pt):
        return .5*(pt*Q.dot(pt)).sum() - (b*pt).sum()
    #pts = [np.array([1.5,1.5]), np.array([1.,1.]), np.array([.5,.5])]
    x0 = np.array([5.,.5])
    pts = steepestDescent(Q, b, x0, niter=20)
    Q = np.array([[1.,0], [0,10.]])
    b = np.array([0.,0.])
    x0 = np.array([5.,.5])
    pts2 = conjugateGradient(Q, b, x0)
    dom = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(dom, dom)
    Z = np.empty(X.shape)
    for i in xrange(X.shape[0]):
        for j in xrange(X.shape[1]):
            pt = np.array([X[i,j], Y[i,j]])
            Z[i,j] = f(pt)
    vals = np.empty(len(pts))
    for i in xrange(len(pts)):
        vals[i] = f(pts[i])
    plt.contour(X,Y,Z, vals[:5], colors='gray')
    plt.plot(np.array(pts)[:,0],np.array(pts)[:,1], '*-')
    plt.plot(np.array(pts2)[:,0],np.array(pts2)[:,1], '*-')
    plt.savefig('steepVsConj.pdf')
    plt.clf()


def linReg():
    # Do a simple linear regression with one predictor variable
    x = np.random.random(10)*10
    y = 2.5*x + np.random.randn(10)
    
    A = np.ones((10,2))
    A[:,1] = x
    Q = A.T.dot(A)
    b = A.T.dot(y)
    x0 = np.array([1, 1.])
    x1 = conjugateGradient(Q,b,x0,allpts=False)
    
    dom = np.linspace(0,10,2)
    plt.plot(dom, x1[0]+x1[1]*dom)
    plt.scatter(x,y)
    plt.savefig('linregression.pdf')
    plt.clf()


def logReg():
    def objective(b):
        '''Return -1*l(b[0], b[1]), where l is the log likelihood.'''
        return np.log(1+np.exp(b[0]+b[1]*x)).sum() - (y*(b[0]+b[1]*x)).sum()
    
    y = np.array([0, 0, 0, 0, 1, 0, 1, 0, 1, 1])
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    guess = np.array([1., 1.])
    b = fmin_cg(objective, guess)
    dom = np.linspace(0,11,100)
    plt.plot(x, y, 'o')
    plt.plot(dom, 1./(1+np.exp(-b[0]-b[1]*dom)))
    plt.savefig('logreg.pdf')
    plt.clf()


if __name__ == "__main__":
    logReg()
    linReg()
    steepVsConj()
