from numpy cimport ndarray as ar

cdef extern from "ctridiag.h":
    void ctridiag(double* a, double* b, double* c, double* x, int n)

def cytridiag(ar[double] a, ar[double] b, ar[double] c, ar[double] x):
    cdef int n = x.size
    ctridiag(&a[0], &b[0], &c[0], &x[0], n)
