# Import the declaration for the C++ Permutation class from
# the Permuation declaration file.
from Permutation cimport Permutation
# Also import the vector template.
# This is a necessary part of the interface for the Permuation class.
from libcpp.vector cimport vector

# Shorten the type identifier for unsigned integers.
# This is just for convenience in the code.
ctypedef unsigned int uint

# Here we define the Cython wrapper class.
cdef class PyPermutation:
    # Make the wrapper class store a pointer to an instance
    # of the C++ class we are wrapping.
    cdef Permutation* thisptr
    # Here is the initialization step.
    # This is a special Cython function that is
    # called when one of the objects for the wrapper
    # class is initialized.
    # This is where we deal with all necessary initialization.
    # It is somewhat analogous to a constructor for a C++ class.
    def __cinit__(self, li=None):
        cdef vector[vector[uint]] pre_permutation
        cdef vector[uint] pre_cycle
        # Here we perform some basic checking for consistency
        # in the input and also build the vectors that will
        # be used to construct the Permutation object.
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
    # This is another special Cython function that is
    # called when one of these wrapper classes is deallocated.
    # Here we take care of all necessary deallocation steps.
    # In this case, all necessary deallocation is taken care of
    # by the destructor method for the Permutation object,
    # so all we have to do is use the del operator to deallocate it.
    def __dealloc__(self):
        del self.thisptr
    # Here we define how the object is printed.
    # Notice the automatic conversion from a
    # C++ string to a Python string.
    def __repr__(self):
        return self.thisptr.get_string()
    # This returns the index of a given index under the permutation.
    def trace(self, index):
        if index < 0:
            raise ValueError("index must be nonnegative")
        return self.thisptr.trace(index)
    # This returns the preimage of a given index under the permutation.
    def trace_inverse(self, index):
        if index < 0:
            raise ValueError("index must be nonnegative")
        return self.thisptr.trace_inverse(index)
    # This computes and returns a PyPermutation object that wraps
    # a C++ Permutation object representing the inverse of this permutation.
    def inverse(self):
        inv = PyPermutation()
        inv.thisptr = self.thisptr.inverse()
        return inv
    # Here we define some basic methods for viewing attributes of
    # the Permutation object.
    # This returns the maximum index modified by the permutation.
    def get_max(self):
        return self.thisptr.get_max()
    # This returns the minimum index modified by the permutation.
    def get_min(self):
        return self.thisptr.get_min()
    # This returns the number of cycles currently stored in the permutation.
    def get_size(self):
        return self.thisptr.get_size()
    # This method reduces the Permutation to its disjoint cycle form.
    def reduce(self):
        self.thisptr.reduce()
    # This method performs multiplication between two Permutation objects.
    def __mul__(PyPermutation self, PyPermutation other):
        # Raise an error if the user tries to multiply this object by
        # anything other than another PyPermutation object.
        if not isinstance(other, PyPermutation):
            raise NotImplementedError("Can only multiply Permutations by other permutations")
        # Start by initializing a new PyPermutation object
        # containing a null pointer.
        cdef PyPermutation result = PyPermutation()
        # Set its 'thisptr' attribute equal to a pointer
        # to the result of multiplication between the two
        # Permutation objectst we are multiplying.
        # Notice that we dereference the pointers to perform
        # the multiplication, but that, as defined in the Permutation header,
        # the multiplication operator returns a pointer to the result.
        result.thisptr = (self.thisptr[0]) * (other.thisptr[0])
        # Return the PyPermutation object we have just constructed.
        return result
    def __pow__(PyPermutation self, int p, z):
        # 'z' is a required part of the calling signature for this function in Cython.
        # It is used in taking powers of integers using modular arithmetic.
        # Here we raise an error if it is not None since we have not
        # defined modular arithmetic for permutation objects.
        if z is not None:
            raise NotImplementedError("modular arithmetic with permutations is not defined")
        # As we did in the multiplication method,
        # we construct an empty PyPermutation object.
        cdef PyPermutation result = PyPermutation()
        # The power method also returns a pointer, so we dereference
        # the pointer, call the method, and assign the pointer
        # in the new PyPermutation object to point to the result of the computation.
        result.thisptr = (self.thisptr[0]).power(p)
        # Now we return the new PyPermutation object.
        return result
