# Include the NumPy C headers to interface with arrays.
# Alias the array type as "ar".
from numpy cimport ndarray as ar

# Include the ctridiag function as declared in the header file.
# Define the interface here as well.
# Use the Cython syntax instead of the C syntax.
cdef extern from "ctridiag.h":
    void ctridiag(double* a, double* b, double* c, double* x, int n)

# Define a Python wrapper for the ctridiag function.
# Take four NumPy arrays as input, verify that they are arrays
# of double precision floating point values.
# Pass the size of 'x' to ctridiag to specify the
# sizes of all the arrays
# Warning! This will not check for out of bounds memory accesses.
# Either make sure the arrays you pass it are all the proper sizes
# or add the checking here before you call the C function.
cpdef cytridiag(ar[double] a, ar[double] b, ar[double] c, ar[double] x):
    cdef int n = x.size
    # Notice that we pass pointers to the first element
    # of each array instead of the arrays themselves.
    ctridiag(&a[0], &b[0], &c[0], &x[0], n)
