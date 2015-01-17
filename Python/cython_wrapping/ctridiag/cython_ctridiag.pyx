# Include the ctridiag() function as declared in ctridiag.h
# We use a cdef since this function will only be called from Cython
cdef extern from "ctridiag.h":
    void ctridiag(double* a, double* b, double* c, double* x, int n)

# Define a Cython wrapper for ctridiag().
# Accept four NumPy arrays of doubles
# This will not check for out of bounds memory accesses.
cpdef cytridiag(double[:] a, double[:] b, double[:] c, double[:] x):
    cdef int n = x.size
    ctridiag(&a[0], &b[0], &c[0], &x[0], n)
