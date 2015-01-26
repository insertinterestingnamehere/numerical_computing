import numpy as np
from numpy cimport ndarray as ar
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline dot(double[:] A, double[:] B, int n):
    cdef double tot=0.
    cdef int i
    for i in xrange(n):
        tot += A[i] * B[i]
    return tot

@cython.boundscheck(False)
@cython.wraparound(False)
def cyrowdot(ar[double, ndim=2] A):
    cdef ar[double, ndim=2] B = np.empty((A.shape[0], A.shape[0]))
    cdef double[:,:] Aview = A
    cdef double temp
    cdef int i, j, h=A.shape[0], w=A.shape[1]
    for i in xrange(h):
        for j in xrange(i):
            temp = dot(Aview[i], Aview[j], w)
            B[i,j] = temp
        B[i,i] = dot(Aview[i], Aview[i], w)
    for i in xrange(h):
        for j in xrange(i+1,h):
            B[i,j] = B[j,i]
    return B
