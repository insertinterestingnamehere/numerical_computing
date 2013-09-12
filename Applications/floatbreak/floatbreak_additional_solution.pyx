from libc.math cimport sqrt

# Note: if you want you can replace the unsigned long long types with
# np.uint64_t types after cimporting numpy.
# This is a pure cython version.
def cy_find_triple_correct(unsigned long long start, unsigned long long num):
    cdef unsigned long long i, j
    cdef double temp
    for i in xrange(num):
        for j in xrange(i, num):
            temp = (start+i)**2 + (start+j)**2
            temp = sqrt(temp)
            if int(temp+1)**2 - (start+i)**2 - (start+j)**2 == 0:
                print start+i, start+j, int(temp+1)
            if int(temp)**2 - (start+i)**2 - (start+j)**2 == 0:
                print start+i, start+j, int(temp)
