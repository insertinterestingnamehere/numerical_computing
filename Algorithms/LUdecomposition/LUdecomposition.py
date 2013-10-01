import numpy as np
from scipy import linalg as la

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
    '''
    this same as the LU function above, just replace the 
    L's and U's by A. Also, we need to make sure to not
    set the value of entries below the main diagonal twice.
    ''
    for j in xrange(A.shape[0]-1):
        for i in xrange(j+1, A.shape[0]):
			# change to L
            A[i,j] /= A[j,j]    
                # we just set entries of jth column below the main diagonal
            # change to U
            A[i,j+1:] -= A[i,j] * A[j,j+1:] 
                # start from column j+1 to avoid setting entires in jth column again

def Solve():
    # the students should have code to generate the random matrices, inverse, LU, and solve
    A = np.random.rand(1000,1000)
    B = np.random.rand(1000,500)
    Ai = la.inv(A)
    (lu,piv) = la.lu_factor(A)
    
    # the students should report the time for the following operations
    np.dot(Ai,B)
    la.lu_solve((lu,piv),B)



def LU_solve(A,B):
    for j in xrange(A.shape[0]-1):
        for i in xrange(j+1, A.shape[0]):
            B[i] -= A[i,j] * B[j]
    for j in xrange(A.shape[0]-1, -1, -1):
        B[j] /= A[j,j]
        for i in xrange(j):
            B[i] -= A[i,j] * B[j]

def LU_det(A):
    (lu,piv) = la.lu_factor(A)
    
    # determine whether an even or odd number of row swaps
    s = (piv != np.arange(A.shape[0])).sum() % 2
    
    return ((-1)**s) * lu.diagonal().prod()

def cholesky(A):
    L = np.zeros_like(A)
    for i in xrange(A.shape[0]):
        for j in xrange(i):
            L[i,j]=(A[i,j] - np.inner(L[i,:j], L[j,:j])) / L[j,j]
        sl = L[i,:i]
        L[i,i] = sqrt(A[i,i] - np.inner(sl, sl))
    return L

def cholesky_inplace(A):
    for i in xrange(A.shape[0]):
        A[i,i+1:] = 0.
        for j in range(i):
            A[i,j] = (A[i,j] - np.inner(A[i,:j],A[j,:j])) / A[j,j]
        sl = A[i,:i]
        A[i,i] = sqrt(A[i,i] - np.inner(sl, sl))

def cholesky_solve(A, B):
    for j in xrange(A.shape[0]):
        B[j] /= A[j,j]
        for i in xrange(j+1, A.shape[0]):
            B[i] -= A[i,j] * B[j]
    for j in xrange(A.shape[0]-1, -1, -1):
        B[j] /= A[j,j]
        for i in xrange(j):
            B[i] -= A[j,i] * B[j]

