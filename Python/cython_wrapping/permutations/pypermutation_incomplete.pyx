from Permutation cimport Permutation
from libcpp.vector cimport vector

ctypedef unsigned int uint

# TODO make sure the built in deallocation method properly deallocates
# the permutation.

cdef class PyPermutation:
    cdef Permutation* thisptr
    def __cinit__(self, li=None):
        cdef vector[vector[uint]] pre_permutation
        cdef vector[uint] pre_cycle
        if li is not None:
            # works assuming li is a list of lists of ints.
            for cycle in li:
                for index in cycle:
                    if index < 0:
                        raise ValueError("index must be nonnegative")
                    pre_cycle.push_back(index)
                pre_permutation.push_back(pre_cycle)
                pre_cycle.clear()
            self.thisptr = new Permutation(pre_permutation)
            # add verify method for Permutation.
            # verify here.
            # then reduce.
    def __dealloc__(self):
        del self.thisptr
    def __repr__(self):
        return self.thisptr.get_string()
    def trace(self, index):
        if index < 0:
            raise ValueError("index must be nonnegative")
        return self.thisptr.trace(index)
    # def trace_inverse(self, index):
    def inverse(self):
        inv = PyPermutation()
        inv.thisptr = self.thisptr.inverse()
        return inv
    def get_max(self):
        return self.thisptr.get_max()
    # def get_min(self):
    def get_size(self):
        return self.thisptr.get_size()
    def reduce(self):
        self.thisptr.reduce()
    def __mul__(PyPermutation self, PyPermutation other):
        if not isinstance(other, PyPermutation):
            raise NotImplementedError("Can only multiply Permutations by other permutations")
        cdef PyPermutation result = PyPermutation()
        result.thisptr = (self.thisptr[0]) * (other.thisptr[0])
        return result
    #def __pow__(PyPermutation self, int p, z):
