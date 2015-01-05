from numpy cimport ndarray as ar
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def cydot(ar[double] A, ar[double] B):
    cdef int i
    cdef double tot = 0.
    for i in xrange(A.size):
        tot += A[i] * B[i]
    return tot
