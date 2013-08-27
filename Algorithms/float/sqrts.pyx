from numpy cimport ndarray as ar
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
def cyinvsqrt32(ar[np.float32_t] A, int reps):
    cdef ar[np.float32_t] Ac = A.copy()
    cdef ar[np.int32_t] I = Ac.view(dtype=np.int32)
    cdef int i, j
    cdef np.int32_t c = 0x5f3759df
    cdef np.float32_t temp, half=.5, threehalves=1.5
    for i in xrange(A.size):
        temp = A[i] * half
        I[i]  = c - ((I[i]) >> 1)
        for j in xrange(reps):
            Ac[i] = Ac[i] * (threehalves - temp * Ac[i] * Ac[i])
    return Ac

@cython.boundscheck(False)
@cython.wraparound(False)
def cyinvsqrt64(ar[double] A, int reps):
    cdef ar[double] Ac = A.copy()
    cdef ar[np.int64_t] I = Ac.view(dtype=np.int64)
    cdef np.int64_t c = 0x5fe6ec85e7de30da
    cdef int i, j
    cdef double temp, half=.5, threehalves=1.5
    for i in xrange(A.size):
        temp = A[i] * half
        I[i] = c - ((I[i]) >> 1)
        for j in xrange(reps):
            Ac[i] = Ac[i] * (threehalves - temp * Ac[i] * Ac[i])
    return Ac

@cython.boundscheck(False)
@cython.wraparound(False)
def cinvsqrt32(ar[np.float32_t] A):
    cdef int i
    cdef ar[np.float32_t] res = np.empty_like(A)
    for i in xrange(A.size):
        res[i] = 1 / sqrt(A[i])
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
def cinvsqrt64(ar[double] A):
    cdef int i
    cdef ar[double] res = np.empty_like(A)
    for i in xrange(A.size):
        res[i] = 1 / sqrt(A[i])
    return res
