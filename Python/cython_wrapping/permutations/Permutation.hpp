#pragma once
#include "Cycle.hpp"
#include "Node.hpp"
#include <list>
#include <vector>
#include <string>
#include <sstream>
#include <set>

// TODO add thorough comments.

// TODO use default constructor when possible.

// TODO Check for length 0 stuff in print reduce method and multiply method.
//  be sure to check that max and min are set properly in those cases.
//  Really this needs to be well-defined behaviour throughout these classes.

// TODO Fix spacing and this-> arrows throughout.

// TODO switch membership test outside of do-while loops in reduce and operator*

// TODO size argument isn't necessary anymore. remove it.

// TODO update minimum and maximum during reduction and not at the end.
//  remember to leave the check for length 0 permutations.

// TODO make is_reduced flag or reduce on construction. At least deal with that case somehow.
//  In particular, the inverse and the inverse trace routines will
//  not work if the Permutation object is not reduced.

class Permutation{
    public:
        Permutation();
        Permutation(std::vector<std::vector<unsigned int> >& cycles);
        ~Permutation();
        Permutation(const Permutation& other);
        Permutation& operator=(const Permutation& other);
        bool verify();
        std::string get_string();
        unsigned int trace(unsigned int index);
        unsigned int trace_inverse(unsigned int index);
        Permutation* inverse();
        unsigned int get_max();
        unsigned int get_min();
        unsigned int get_size();
        //bool index_in(unsigned int index);
        void reduce();
        Permutation* operator*(Permutation& other);
        Permutation* power(int power);
        
        //template <class arr_type>
        //void Permutation::apply(arr_type ar, unsigned int size, int& info){}
        
        //template <class arr_type>
        //void Permutation::apply_inverse(arr_type ar, unsigned int size, int& info){}
        
    private:
        // Require permutations to be sorted and reduced in all public constructors.
        Permutation(std::vector<Cycle*>& cycles, unsigned int size, unsigned int minimum, unsigned int maximum);
        void sort();
        
        // !!!!! This needs to be a list of pointers to avoid memory leaks.
        
        std::vector<Cycle*> cycles;
        unsigned int size;
        unsigned int minimum;
        unsigned int maximum;};
        
