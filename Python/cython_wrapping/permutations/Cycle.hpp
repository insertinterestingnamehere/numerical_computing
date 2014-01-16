#pragma once
#include "Node.hpp"
#include <list>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>

class Permutation;

// TODO: Make sure when constructing new versions that the default constructor is
// used instead of manually constructing the vector, etc.

// TODO: check corner cases for zero-length construction, etc.

// TODO: remove unnecessary constructors.

// TODO: redo comments.

// TODO: fix -1 increment in indices (Maybe on the Python end?? in the constructor?)

class Cycle{
    friend class Permutation;
    public:
        Cycle(std::list<unsigned int>& indexlist);
        Cycle(std::vector<unsigned int>& indexvector);
        Cycle(unsigned int* indexarr, unsigned int size);
        Cycle(const Cycle& other);
        ~Cycle();
        Cycle& operator=(const Cycle& other);
        std::string get_string();
        unsigned int get_min();
        unsigned int get_max();
        unsigned int get_size();
        bool in(unsigned int index);
        unsigned int trace(unsigned int index);
        Cycle* inverse();
        unsigned int trace_inverse(unsigned int index);
        bool operator<(const Cycle& other);
        bool operator>(const Cycle& other);
        bool operator<=(const Cycle& other);
        bool operator>=(const Cycle& other);
        bool operator==(const Cycle& other);
        bool operator!=(const Cycle& other);
        template <class arr_type>
        void apply(arr_type* arr, unsigned int size, int& info){
            if ((this->maximum)<info){
                Node* iter = (this->nodes)[0]->next;
                arr_type temp1 = arr[this->minimum];
                arr_type temp2 = temp1;
                while (iter != (this->nodes)[0]){
                    temp1 = arr[iter->index];
                    arr[iter->index] = temp2;
                    temp2 = temp1;
                    iter = iter->next;}
                info = 0;}
            else{ info = 1;}}
        template <class arr_type>
        void apply_inverse(arr_type* arr, unsigned int size, int& info){
            if ((this->maximum)<info){
                Node* iter = (this->nodes)[0]->previous;
                arr_type temp1 = arr[this->minimum];
                arr_type temp2 = temp1;
                while (iter != (this->nodes)[0]){
                    temp1 = arr[iter->index];
                    arr[iter->index] = temp2;
                    temp2 = temp1;
                    iter = iter->previous;}
                info = 0;}
            else{ info = 1;}}
    private:
        unsigned int minimum;
        unsigned int maximum;
        unsigned int size;
        std::vector<Node*> nodes;
        //Cycle(std::vector<Node*> nodearr, unsigned int size);
        Cycle(std::list<unsigned int>& indexlist, unsigned int size);
        Cycle(std::vector<Node*>& nodearr, unsigned int size, unsigned int minimum, unsigned int maximum);
        bool verify();
        void sort();};
