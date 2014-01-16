from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
ctypedef unsigned int uint
cdef extern from "Permutation.hpp":
    cdef cppclass Permutation:
        Permutation() nogil except +
        Permutation(vector[vector[uint]]& cycles) nogil except +
        bool verify() nogil
        string get_string() nogil
        uint trace(uint index) nogil
        Permutation* inverse() nogil
        uint get_max() nogil
        uint get_size() nogil
        void reduce() nogil
        Permutation* operator*(Permutation& other) nogil
