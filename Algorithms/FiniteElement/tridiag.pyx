from numpy cimport ndarray as arr
from libc.math cimport fabs, sin, cos

def tridiag(arr[double] a, arr[double] b, arr[double] c, arr[double] x): 
	#note: overrides c and x 
	cdef int size = x.size
	cdef int n
	cdef int b_size = b.size
	size = x.size 
	cdef double temp = 0. 
	size = x.size
	c[0] = c[0] / b[0] 
	x[0] = x[0] / b[0] 
	for n in range(size-2): 
		temp = 1. / (b[n+1] - a[n]*c[n]) 
		c[n+1] *= temp 
		x[n+1] = (x[n+1] - a[n]*x[n]) * temp 
	x[size-1] = (x[size-1] - a[size-2]*x[size-2]) / (b[size-1] - a[size-2]*c[size-2]) 
	for n in range(b_size-2, -1, -1): 
		x[n] = x[n] - c[n] * x[n+1] 
	return x