import numpy as np
import math
from scipy import linalg as la

def gmres(Amul,b, k=100, tol=1e-8):
    """
    Calculate approximate solution of Ax = b using GMRES algorithm.
    Inputs:
        Amul -- callable function that calculates Ax for any input vector x.
        b -- numpy array of length m
        k -- max number of iterations to run
        tol -- threshold for detecting convergence
    Returns:
        x -- numpy array of length m, the appoximate solution
        res -- float, giving the residual
    """
    # initialization steps
    m = b.size
    Q = np.empty((m,k))
    H = np.zeros((k+1,k))
    bnorm = la.norm(b,2)
    rhs = np.zeros(k+1)
    rhs[0] = bnorm
    Q[:,0] = b/bnorm

    for j in xrange(k-1):
        # Arnoldi iteration
        q = Amul(Q[:,j])
        for i in xrange(j+1):
            H[i,j] = np.inner(Q[:,i],q)
            q -= H[i,j]*Q[:,i]
        H[j+1,j] = la.norm(q,2)
        if H[j+1,j] > 1e-10:
            # don't divide by zero!
            q /= H[j+1,j]
        Q[:,j+1] = q

        # solve the least squares problem
        y, r = la.lstsq(H[:j+2,:j+1], rhs[:j+2])[:2]

        # compute the residual.
        r = math.sqrt(r)/bnorm
        if r < tol:
            # if we are sufficiently close to solution, return
            return Q[:,:j+1].dot(y), r
    return Q[:,:j+1].dot(y.flatten()), r
