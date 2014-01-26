using namespace std;
#include "Permutation.hpp"
#include <vector>
#include <string>
#include <iostream>

int main(){
    vector<vector<unsigned int> > pre_permutation;
    // first cycle:
    vector<unsigned int> first_cycle;
    first_cycle.push_back(1);
    first_cycle.push_back(5);
    first_cycle.push_back(8);
    first_cycle.push_back(6);
    first_cycle.push_back(4);
    pre_permutation.push_back(first_cycle);
    // second cycle:
    vector<unsigned int> second_cycle;
    second_cycle.push_back(5);
    second_cycle.push_back(12);
    second_cycle.push_back(9);
    second_cycle.push_back(6);
    pre_permutation.push_back(second_cycle);
    // third cycle:
    vector<unsigned int> third_cycle;
    third_cycle.push_back(13);
    third_cycle.push_back(2);
    third_cycle.push_back(12);
    third_cycle.push_back(1);
    third_cycle.push_back(3);
    pre_permutation.push_back(third_cycle);
    Permutation P = Permutation(pre_permutation);
    cout << "before reducing: " << P.get_string() << endl;
    P.reduce();
    cout << "after reducing: " << P.get_string() << endl;
    // Correct output after reducing:
    // (2~12~9~6~4~3~13)(5~8)
    Permutation* PP = P*P;
    cout << PP->get_string() << endl;
    delete PP;
    Permutation* P2 = P.power(2);
    cout << P2->get_string() << endl;
    delete P2;
    system("pause");}
