import numpy as np
    
def ref(A):
    for i in xrange(A.shape[0]-1):
        for j in xrange(i+1, A.shape[0]):
            A[j,i:] -= (A[j,i] / A[i,i]) * A[i,i:]

def LU(A):
    U = A.copy()
    L = np.eye(A.shape[0])
    for i in xrange(A.shape[0]-1):
        for j in xrange(i+1, A.shape[0]):
            #operation corresponding to left mult by
            #the elementary matrix desired
            L[j,i] = (U[j,i] / U[i,i])
            #now we apply the change to U
            U[j,i:] -= L[j,i] * U[i,i:]
    return L, U

def LU_inplace(A):
    for i in xrange(A.shape[0]-1):
        for j in xrange(i+1,A.shape[0]):
			#change to L
            A[j,i] /= A[i,i]
            #change to U
            A[j,i+1:] -= A[j,i] * A[i,i+1:]

def LU_solve(A,B):
    for i in xrange(A.shape[0]-1):
        for j in xrange(i+1,A.shape[0]):
            B[j] -= A[j,i] * B[i]
    for i in xrange(A.shape[0]-1,-1,-1):
        B[i] /= A[i,i]
        for j in xrange(i):
            B[j] -= A[j,i] * B[i]

def LU_det(A):
    B = A.copy()
    ref(B)
    #now extract diagonal and take product
    return np.prod(B.reshape(B.size)[::B.shape[1]+1])
