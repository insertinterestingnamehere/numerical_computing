import numpy as np
from scipy import linalg

def QR(X):
    """Compute the Gram Schmidt of the column vectors in X

    The formula: x_k := x_k-<x_k, q_1>*q_1  (k=2,...,n)
    Then we normalize x_k"""

    #transpose so we are dealing with rows instead of columns
    #check types on X
    Q = X.T.copy()
    nrows, ncols = X.shape
    R = np.zeros((nrows, ncols))

    for i in xrange(nrows):
        R[i,i] = linalg.norm(Q[i])
        Q[i] = Q[i]/R[i,i]
        for j in xrange(i+1,nrows):
            R[i,j] = Q[j].dot(Q[i])
            Q[j] = Q[j]-(R[i,j]*Q[i])

    return Q.T, R
    
    
def detQR(X):
    """Computes the determinant of X using the QR decomposition
    This will give you the determinant up to, but without sign.
    
    The determinant of Q is +1 or -1 which may or may not change the sign of
    the determinant of R"""
    Q, R = QR(X)
    return np.diagonal(R).prod()

def leastsq(A, b):
    Q, R = linalg.qr(A)
    return linalg.solve_triangular(R, Q.T.dot(b))
    
    
def eigvv(A, niter=50):
    Qlist = [A]
    x0 = np.random.rand(A.shape[1])
    for i in range(niter):
        Q,R = QR(Qlist[-1])
        A = Q.T.dot(A.dot(Q))        
        Qlist.append(A)
    
    eigvals = np.diag(Qlist[-1])
    
    L = np.eye(Q.shape[0])    
    for qm in Qlist:
        L = L.dot(qm)
       
    return eigvals, x0, L
