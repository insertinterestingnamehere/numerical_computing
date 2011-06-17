import scipy as sp
from scipy.linalg import norm

def Q(X):
    """Compute the Gram Schmidt of the column vectors in X

    The formula: x_k := x_k-<x_k, q_1>*q_1  (k=2,...,n)
    Then we normalize x_k"""

    #transpose so we are dealing with rows instead of columns
    Q = X.T.copy().astype('float64')
    nrows, ncols = Q.shape
    R = sp.zeros((nrows, ncols))

    for i in range(nrows):
        R[i,i] = norm(Q[i])
        Q[i] = Q[i]/R[i,i]
        for j in range(i+1,nrows):
            R[i,j] = sp.dot(Q[j], Q[i])
            Q[j] = Q[j]-(R[i,j]*Q[i])

    return Q.T, R
