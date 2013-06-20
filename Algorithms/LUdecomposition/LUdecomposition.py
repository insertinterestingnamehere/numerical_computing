import numpy as np
    
def ref(A):
    for j in xrange(A.shape[0]-1):
        for i in xrange(j+1, A.shape[0]):
            A[i,j:] -= (A[i,j] / A[j,j]) * A[j,j:]

def LU(A):
    U = A.copy()
    L = np.eye(A.shape[0])
    for j in xrange(A.shape[0]-1):
        for i in xrange(j+1, A.shape[0]):
            #operation corresponding to left mult by
            #the elementary matrix desired
            L[i,j] = (U[i,j] / U[j,j])
            #now we apply the change to U
            U[i,j:] -= L[i,j] * U[j,j:]
    return L, U

def LU_inplace(A):
    for j in xrange(A.shape[0]-1):
        for i in xrange(j+1,A.shape[0]):
			#change to L
            A[i,j] /= A[j,j]
            #change to U
            A[i,j+1:] -= A[i,j] * A[j,j+1:]

def LU_solve(A,B):
    for j in xrange(A.shape[0]-1):
        for i in xrange(j+1,A.shape[0]):
            B[i] -= A[i,j] * B[j]
    for j in xrange(A.shape[0]-1,-1,-1):
        B[j] /= A[j,j]
        for i in xrange(j):
            B[i] -= A[i,j] * B[j]

def LU_det(A):
    B = A.copy()
    ref(B)
    #now extract diagonal and take product
    return np.prod(B.reshape(B.size)[::B.shape[1]+1])
