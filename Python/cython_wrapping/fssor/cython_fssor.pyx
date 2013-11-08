from numpy cimport ndarray as ar

cdef extern from "fssor.h":
    void fssor(double* U, int* m, int* n, double* omega, double* tol, int* maxiters, int* info)

def cyssor(ar[double,ndim=2] U, double omega, double tol=1E-8, int maxiters=1000):
    cdef int m=U.shape[0], n=U.shape[1], info
    if U.flags["F_CONTIGUOUS"]:
        fssor(&U[0,0], &m, &n, &omega, &tol, &maxiters, &info)
    elif U.flags["C_CONTIGUOUS"]:
        fssor(&U[0,0], &n, &m, &omega, &tol, &maxiters, &info)
    else:
        raise ValueError("Array must be either C or Fortran contiguous")
    if info:
        raise ValueError("Failed to converge within given tolerance")
