# Import the needed classes from the standard template library.
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

# Shorten the type identifier for unsigned integers.
# This is just for convenience in the code.
ctypedef unsigned int uint

# Here we declare that we will be importing external
# functions and/or classes from the header "Permutation.hpp".
cdef extern from "Permutation.hpp":
    # This line declares that we are importing a C++ class.
    cdef cppclass Permutation:
        # Here we define the constructors.
        # We use the nogil keyword to tell Cython that
        # the global interpreter lock should not apply to these
        # functions.
        # The constructor method also needs the extra keyword 'except +'.
        # This tells Cython to raise a proper Python exception if
        # a C++ error is raised.
        Permutation() nogil except +
        Permutation(vector[vector[uint]]& cycles) nogil except +
        # From here on we declare the other methods we may be
        # interested in using in Cython.
        bool verify() nogil
        string get_string() nogil
        uint trace(uint index) nogil
        Permutation* inverse() nogil
        uint get_max() nogil
        uint get_size() nogil
        void reduce() nogil
        Permutation* operator*(Permutation& other) nogil
