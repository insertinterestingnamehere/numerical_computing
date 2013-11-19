from numpy cimport ndarray as ar

cdef extern from "cssor.h":
    void cssor(double* U, unsigned int m, unsigned int n, double omega, double tol, unsigned int maxiters, unsigned int info)

def cyssor(ar[double,ndim=2] U, double omega, double tol=1E-8, unsigned int maxiters=10000):
    cdef unsigned int m=U.shape[0], n=U.shape[1], info
    if U.flags["F_CONTIGUOUS"]:
        # If U is Fortran contiguous, call it exactly as it is written.
        cssor(&U[0,0], m, n, omega, tol, maxiters, info)
    elif U.flags["C_CONTIGUOUS"]:
        # If it is C contiguous, call it with the axes swapped.
        # The algorithm will still converge fine.
        cssor(&U[0,0], n, m, omega, tol, maxiters, info)
    else:
        # The Fortran algorithm we have written is not general enough
        # to handle non-contiguous arrays. Raise an error.
        raise ValueError("Array must be either C or Fortran contiguous")
    if info:
        # Raise this error if it failed to converge.
        raise ValueError("Failed to converge within given tolerance")
