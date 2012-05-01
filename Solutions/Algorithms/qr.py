import scipy as sp
from scipy.linalg import norm, det

def QR(X):
    """Compute the Gram Schmidt of the column vectors in X

    The formula: x_k := x_k-<x_k, q_1>*q_1  (k=2,...,n)
    Then we normalize x_k"""

    #transpose so we are dealing with rows instead of columns
    #check types on X
    Q = X.T.copy()
    nrows, ncols = Q.shape
    R = sp.zeros((nrows, ncols))

    for i in range(nrows):
        R[i,i] = norm(Q[i])
        Q[i] = Q[i]/R[i,i]
        for j in range(i+1,nrows):
            R[i,j] = Q[j].dot(Q[i])
            Q[j] = Q[j]-(R[i,j]*Q[i])

    return Q.T, R

def detQR(X):
    """Computes the determinant of X using the QR decomposition"""
    Q, R = QR(X)
    return la.det(Q)*la.det(R)
    
def eigvv(A, niter=50):
    Qlist = [A]
    x0 = sp.rand(A.shape[1])
    for i in range(niter):
        Q,R = QR(Qlist[-1])
        A = Q.T.dot(A.dot(Q))        
        Qlist.append(A)
    
    eigvals = sp.diag(Qlist[-1])
    
    L = sp.eye(Q.shape[0])    
    for qm in Qlist:
        L = sp.dot(L,qm)
       
    return eigvals, x0, L
