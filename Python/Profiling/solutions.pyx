import numpy as np
cimport cython


# Technically, putting the Python functions here could speed them up
# or slow them down somewhat because some operations are compiled to
# use the Python C API. It's really simpler to have everything in one
# place though.


# 1D Array Sum Problem

def pysum(X):
    tot = 0.
    for i in xrange(X.size):
        tot += X[i]
    return tot

def cysum1(X):
    cdef:
        int i
        double tot = 0.
    for i in xrange(X.shape[0]):
        tot += X[i]
    return tot

def cysum2(double[:] X):
    cdef:
        int i
        double tot = 0.
    for i in xrange(X.shape[0]):
        tot += X[i]
    return tot

@cython.boundscheck(False)
@cython.wraparound(False)
def cysum3(double[:] X):
    cdef:
        int i
        double tot = 0.
    for i in xrange(X.shape[0]):
        tot += X[i]
    return tot

# Here are two other versions that can be used to show that the array
# speed really is just as good as raw pointer arithmetic.

def cysum4(double[:] X):
    cdef:
        int i
        double tot = 0.
        double *Xptr = &X[0]
    for i in xrange(X.shape[0]):
        tot += Xptr[0]
        Xptr += 1
    return tot

def cysum5(double[:] X):
    cdef:
        int i
        double tot = 0.
        double *Xptr = &X[0]
    for i in xrange(X.shape[0]):
        tot += Xptr[i]
    return tot

# The plain Python version should be very slow.
# The version with only the typed for loop will be significantly slower.
# The version with typed arrays should be very fast.
# The version with typed arrays and compiler directives should be
#  fairly close to the speed of NumPy's sum function, though NumPy
#  should still be just a little faster.
# This example shows that Cython allows easy implimentation of functions
# that are nearly as fast as the hand-tuned C code used throughout numpy.


# LU Decomposition Problem

# This is an in-place LU decomposition.
# The important idea here is that the students applied similar
# modifications to their own code. They should unroll the loops,
# define the types for the arrays and indices, and use the compiler
# directives to turn off the different indexing checks.

def pylu(A):
    for j in xrange(A.shape[0]-1):
        for i in xrange(j+1,A.shape[0]):
            #change to L
            A[i,j] /= A[j,j]
            #change to U
            A[i,j+1:] -= A[i,j] * A[j,j+1:]

def pylu_slow(A):
    for j in xrange(A.shape[0]-1):
        for i in xrange(j+1,A.shape[0]):
            #change to L
            A[i,j] /= A[j,j]
            #change to U
            for k in xrange(j+1,A.shape[1]):
                A[i,k] -= A[i,j] * A[j,k]

@cython.boundscheck(False)
@cython.wraparound(False)
def cylu(double[:,:] A):
    cdef int i,j,k
    for j in xrange(A.shape[0]-1):
        for i in xrange(j+1,A.shape[0]):
            #change to L
            A[i,j] /= A[j,j]
            #change to U
            for k in xrange(j+1,A.shape[1]):
                A[i,k] -= A[i,j] * A[j,k]

# For large arrays, the Cython version should be markedly faster than
# the Python one, though not spectacularly so because of NumPy's
# vectorization. This example shows both how good Cython's loops on
# typed arrays are, but also how well NumPy's vectorization works for
# large arrays. If anyone bothers to write a version in Python with all
# the for loops done explicitly, it should be spectacularly slow.


# Matrix Power Problem

def pymatpow(X, power):
    prod = X.copy()
    temparr = np.empty_like(X[0])
    size = X.shape[0]
    for n in xrange(1, power):
        for i in xrange(X.shape[0]):
            for j in xrange(X.shape[1]):
                tot = 0.
                for k in xrange(size):
                    tot += prod[i,k] * X[k,j]
                temparr[j] = tot
            prod[i] = temparr
    return prod

@cython.boundscheck(False)
@cython.wraparound(False)
def cymatpow(double[:,:] X, int power):
    cdef:
        double[:,:] prod = X.copy()
        double[:] temparr = np.empty_like(X[0])
        int i, j, k, n, size=X.shape[0]
        double tot
    for n in xrange(1, power):
        for i in xrange(X.shape[0]):
            for j in xrange(X.shape[1]):
                tot = 0.
                for k in xrange(size):
                    tot += prod[i,k] * X[k,j]
                temparr[j] = tot
            # In theory, this copying operation should
            # be unrolled as a loop, but, in this case, it
            # doesn't seem to make much of a difference.
            prod[i] = temparr
    return np.array(prod)

# The Cython version should be dramatically faster than the version
# with the explicit loops in Python. This problem gives them extra
# practice porting an algorithm to Cython and also demonstrates just how
# slow iterating directly over an array really is.
# This problem also illustrates how, in spite of the fact that loops in
# Cython are well-optimized, the built in BLAS and LAPACK routines for
# the various linear algebra operations in numpy are faster than any
# sort of direct looping-based approach.


# Tridiagonal Algorithm Problem

def pytridiag(a, b, c, x):
    # Note: overwrites c and x.
    n = x.size
    temp = 0.
    c[0] /= b[0]
    x[0] /= b[0]
    for i in xrange(n-2):
        temp = 1. / (b[i+1] - a[i] * c[i])
        c[i+1] *= temp
        x[i+1] = (x[i+1] - a[i] * x[i]) * temp
    x[n-1] = (x[n-1] - a[n-2] * x[n-2]) / (b[n-1] - a[n-2] * c[n-2])
    for i in xrange(n-2, -1, -1):
        x[i] = x[i] - c[i] * x[i+1]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cytridiag(double[:] a, double[:] b, double[:] c, double[:] x):
    # Note: overwrites c and x.
    cdef:
        int i, n=x.shape[0]
        double temp
    c[0] /= b[0]
    x[0] /= b[0]
    for i in xrange(n-2):
        temp = 1. / (b[i+1] - a[i] * c[i])
        c[i+1] *= temp
        x[i+1] = (x[i+1] - a[i] * x[i]) * temp
    x[n-1] = (x[n-1] - a[n-2] * x[n-2]) / (b[n-1] - a[n-2] * c[n-2])
    for i in xrange(n-2, -1, -1):
        x[i] = x[i] - c[i] * x[i+1]

# If the students time their code here against the default solver for
# linear systems in scipy, they will see a very dramatic speed increase
# because of the algorithm used. Furthermore, if they compare the speed
# of the Python version of this algorithm and the Cython version, an
# additional speed boost will be apparent. This problem is designed to
# show how both algorithm choice and well-written code contribute to
# an efficient solution. The Cython version of this algorithm should be
# able to solve systems with millions of unknowns in a very short amount
# of time.
