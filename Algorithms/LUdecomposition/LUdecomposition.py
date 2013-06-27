import numpy as np
    
def ref(A):
    for j in xrange(A.shape[0]-1):
        for i in xrange(j+1, A.shape[0]):
            A[i,j:] -= A[i,j] * A[j,j:] / A[j,j]


def ref2(A):
    # This is an alternate version
    # It is faster but less intuitive.
    for i in xrange(A.shape[0]):
        A[i+1:,i:] -= np.outer(A[i+1:,i]/A[i,i], A[i,i:])


def LU(A):
    U = A.copy()
    L = np.eye(A.shape[0])
    for j in xrange(A.shape[0]-1):
        for i in xrange(j+1, A.shape[0]):
            # operation corresponding to left mult by
            # the elementary matrix desired
            L[i,j] = U[i,j] / U[j,j]
            # now we apply the change to U
            U[i,j:] -= L[i,j] * U[j,j:]
    return L, U


def LU2(A):
    # This is an alternate similar to ref2.
    U = A.copy()
    L = np.eye(A.shape[0])
    for i in xrange(A.shape[0]-1):
        L[i+1:,i] = U[i+1:,i] / U[i,i]
        U[i+1:,i:] -= np.outer(L[i+1:,i], U[i,i:])
    return L, U


def LU_inplace(A):
    for j in xrange(A.shape[0]-1):
        for i in xrange(j+1, A.shape[0]):
			# change to L
            A[i,j] /= A[j,j]
            # change to U
            A[i,j+1:] -= A[i,j] * A[j,j+1:]


def LU_solve(A,B):
    for j in xrange(A.shape[0]-1):
        for i in xrange(j+1, A.shape[0]):
            B[i] -= A[i,j] * B[j]
    for j in xrange(A.shape[0]-1, -1, -1):
        B[j] /= A[j,j]
        for i in xrange(j):
            B[i] -= A[i,j] * B[j]


def LU_det(A):
    B = A.copy()
    ref(B)
    # now extract diagonal and take product
    return np.prod(B.diagonal())
