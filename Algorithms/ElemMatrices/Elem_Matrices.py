import numpy as np
    
def ref(A):
    for i in xrange(A.shape[0]):
        for j in xrange(i+1, A.shape[0]):
            A[j] -= (A[j,i] / A[i,i]) * A[i]

def LU(A):
    U = A.copy()
    L = np.eye(A.shape[0])
    for i in xrange(A.shape[0]):
        for j in xrange(i+1, A.shape[0]):
            #operation corresponding to left mult by
            #the elementary matrix desired
            L[:,i] += (U[j,i] / U[i,i]) * L[:,j]
            #now we apply the change to U
            U[j] -= (U[j,i] / U[i,i]) * U[i]
    return L, U

def LU_det(A):
    B = A.copy()
    ref(B)
    #now extract diagonal and take product
    return np.prod(B.reshape(B.size)[::B.shape[1]+1])
