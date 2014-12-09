#include "Permutation.hpp"
#include <list>
#include <vector>
#include <string>
#include <sstream>
#include <set>
#include <algorithm>
#include <iostream>

Permutation::Permutation(){
    // Leave vector uninitialized so it will be empty.
    this->size = 0;
    this->minimum = 0;
    this->maximum = 0;}

Permutation::Permutation(std::vector<std::vector<unsigned int> >& cycles){
    this->size = cycles.size();
    this->cycles.reserve(this->size);
    this->maximum = cycles[0][0];
    this->minimum = cycles[0][0];
    Cycle* c;
    for (std::vector<std::vector<unsigned int> >::iterator i = cycles.begin(); i != cycles.end(); i++){
        c = new Cycle(*i);
        this->cycles.push_back(c);
        this->maximum = (this->maximum < c->maximum) ? c->maximum : this->maximum;
        this->minimum = (this->minimum < c->minimum) ? this->minimum : c->minimum;}}

// Private constructor.
// TODO!!!!!!!! use empty constructor instead of this. It causes unnecessary copying and is less clear.
Permutation::Permutation(std::vector<Cycle*>& cycles, unsigned int size, unsigned int minimum, unsigned int maximum){
    this->size = size;
    this->minimum = minimum;
    this->maximum = maximum;
    this->cycles = cycles;}

Permutation::~Permutation(){
    for (std::vector<Cycle*>::iterator i = this->cycles.begin(); i != this->cycles.end(); i++){
        delete *i;}}

Permutation::Permutation(const Permutation& other){
    this->size = other.size;
    this->minimum = other.minimum;
    this->maximum = other.maximum;
    this->cycles.reserve(this->size);
    for (unsigned int i = 0; i < this->size; i++){
        this->cycles.push_back(new Cycle(*(other.cycles[i])));}}

Permutation& Permutation::operator=(const Permutation& other){
    this->size = other.size;
    this->minimum = other.minimum;
    this->maximum = other.maximum;
    for (unsigned int i = 0; i < this->cycles.size(); i++){
        delete this->cycles[i];}
    this->cycles.reserve(this->size);
    for (unsigned int i = 0; i < other.cycles.size(); i++){
        this->cycles.push_back(new Cycle(*(other.cycles[i])));}}

//struct __cycle_compare{
//    bool operator() (Cycle* i, Cycle* j) {return (*i)<(*j);}};

bool __cycle_compare(Cycle* i, Cycle* j){return (*i)<(*j);}
  
void Permutation::sort(){
    std::sort(this->cycles.begin(), this->cycles.end(), __cycle_compare);}

bool Permutation::verify(){
    for (std::vector<Cycle*>::iterator i=cycles.begin(); i!=cycles.end(); i++){
        if (!((*i)->verify())){
            return false;}}
    return true;}

std::string Permutation::get_string(){
    if (this->size == 0){return "(1)";}
    std::string str;
    std::stringstream stream;
    for (std::vector<Cycle*>::iterator i=(this->cycles).begin(); i!=(this->cycles).end(); i++){
        stream << ((*i)->get_string());}
    stream >> str;
    return str;}

unsigned int Permutation::trace(unsigned int index){
    unsigned int new_index = index;
    for (std::vector<Cycle*>::iterator i=(this->cycles).begin(); i!=(this->cycles).end(); i++){
        new_index = (*i)->trace(new_index);}
    return new_index;}

unsigned int Permutation::trace_inverse(unsigned int index){
    // This only works if the Permutation is already reduced.
    unsigned int new_index = index;
    for (std::vector<Cycle*>::iterator i=(this->cycles).begin(); i!=(this->cycles).end(); i++){
        new_index = (*i)->trace_inverse(new_index);}
    return new_index;}

Permutation* Permutation::inverse(){
    // This only works if the Permutation is already reduced.
    std::vector<Cycle*> new_nodes;
    new_nodes.reserve(this->size);
    for (std::vector<Cycle*>::iterator i=this->cycles.begin(); i!=this->cycles.end(); i++){
        new_nodes.push_back((*i)->inverse());}
    return new Permutation(new_nodes, this->size, this->minimum, this->maximum);}

unsigned int Permutation::get_max(){
    return this->maximum;}

unsigned int Permutation::get_min(){
    return this->minimum;}

unsigned int Permutation::get_size(){
    return this->size;}

// Doesn't currently do any checking that the new cycles are 
// different from the old ones.
// Reducing a Permutation will always result in copying all
// of its cycles even if that isn't always necessary.
void Permutation::reduce(){
    // NOTE: reducing inconsistent Permutations can result in infinite loops.
    std::vector<Cycle*> new_cycles;
    // Note on optimization: The implementation here iterates through
    // the indices in each cycle one cycle at a time.
    // It would probably be better to iterate through
    // The nodes in order by index instead.
    // This would probably require a queue and some sort
    // of a task object.
    // The benefit is that the cycles and permutations wouldn't need
    // to be sorted at the end of iteration.
    std::set<unsigned int> processed;
    std::vector<unsigned int> new_cycle;
    // Iterate over cycles.
    // First get the size of the cycle to be created.
    // Iterate over the cycles.
    unsigned int idx;
    Node* n;
    Cycle* cyc;
    unsigned int mx = 0;
    unsigned int mn = mx - 1;
    // For each cycle in the Permutation:
    for (std::vector<Cycle*>::iterator c = (this->cycles).begin(); c != (this->cycles).end(); c++){
        // Iterate over the indices in each cycle.
        n = (*c)->nodes[0];
        do{
            if (processed.find(n->index)==processed.end()){
                idx = n->index;
                do{
                    new_cycle.push_back(idx);
                    processed.insert(idx);
                    idx = this->trace(idx);
                    }while (idx != (n->index));
                if (new_cycle.size() > 1){
                    cyc = new Cycle(new_cycle);
                    new_cycles.push_back(cyc);
                    if (cyc->minimum < mn){
                        mn = cyc->minimum;}
                    if (cyc->maximum > mx){
                        mx = cyc->maximum;}}
                new_cycle.clear();}
            n = n->next;
            }while (n!=((*c)->nodes[0]));}
    // Delete old nodes.
    for (std::vector<Cycle*>::iterator i = this->cycles.begin(); i != this->cycles.end(); i++){
        delete *i;}
    // Empty the list of cycles.
    this->cycles.clear();
    // Copy in the new pointers.
    this->cycles = new_cycles;
    this->size = this->cycles.size();
    if (this->cycles.size() == 0){
        this->minimum = 0;
        this->maximum = 0;}
    else {
        this->minimum = mn;
        this->maximum = mx;}
    this->sort();}

// Again, there are more efficient ways to do this, but this shouldn't be too bad.
Permutation* Permutation::operator*(Permutation& other){
    Permutation* new_permutation = new Permutation();
    new_permutation->maximum = 0;
    new_permutation->minimum = new_permutation->maximum - 1;
    // As before, it would probably be better to do this with some sort of queue.
    std::set<unsigned int> processed;
    std::vector<unsigned int> new_cycle;
    Node* n;
    Cycle* cyc;
    unsigned int idx;
    for (std::vector<Cycle*>::iterator c = this->cycles.begin(); c != this->cycles.end(); c++){
        n = (*c)->nodes[0];
        do{
            if (processed.find(n->index) == processed.end()){
                idx = n->index;
                do{
                    new_cycle.push_back(idx);
                    processed.insert(idx);
                    idx = this->trace(idx);
                    idx = other.trace(idx);
                    }while (idx != (n->index));
                if (new_cycle.size() > 1){
                    cyc = new Cycle(new_cycle);
                    new_permutation->cycles.push_back(cyc);
                    if (cyc->minimum < new_permutation->minimum){
                        new_permutation->minimum = cyc->minimum;}
                    if (cyc->maximum > new_permutation->maximum){
                        new_permutation->maximum = cyc->maximum;}}
                new_cycle.clear();}
            n = n->next;
            }while (n!=((*c)->nodes[0]));}
    for (std::vector<Cycle*>::iterator c = other.cycles.begin(); c != other.cycles.end(); c++){
        n = (*c)->nodes[0];
        do{
            if (processed.find(n->index) == processed.end()){
                idx = n->index;
                do{
                    new_cycle.push_back(idx);
                    processed.insert(idx);
                    idx = this->trace(idx);
                    idx = other.trace(idx);
                    }while (idx != (n->index));
                if (new_cycle.size() > 1){
                    cyc = new Cycle(new_cycle);
                    new_permutation->cycles.push_back(cyc);
                    if (cyc->minimum < new_permutation->minimum){
                        new_permutation->minimum = cyc->minimum;}
                    if (cyc->maximum > new_permutation->maximum){
                        new_permutation->maximum = cyc->maximum;}}
                new_cycle.clear();}
            n = n->next;
            }while (n!=((*c)->nodes[0]));}
    // size
    new_permutation->size = new_permutation->cycles.size();
    // min/max
    if (new_permutation->cycles.size()){
        new_permutation->minimum = 0;
        new_permutation->maximum = 0;}
    // sort
    new_permutation->sort();
    return new_permutation;}

// gcd needed for next function.
unsigned int __gcd(unsigned int a, unsigned int b){
    unsigned int an, bn;
    if (a <= b){an = b; bn = a;}
    else {an = a; bn = b;}
    unsigned int r = an % bn;
    while (r != 0){
        an = bn;
        bn = r;
        r = an % bn;}
    return bn;}

Permutation* Permutation::power(int power){
    // Cycle must be reduced for this to work properly.
    // We could precompute the space needed in the vector.
    // We could also form groups of cycles separately.
    // Either of the two may or may not be faster.
    Permutation* result = new Permutation();
    if (power == 0){return result;}
    if (power > 0){
        unsigned int p = power;
        unsigned int div;
        std::vector<unsigned int> new_cycle;
        unsigned int new_cycle_size, steps;
        Node *initial_node, *current_node;
        for (std::vector<Cycle*>::iterator c = this->cycles.begin(); c != this->cycles.end(); c++){
            div = __gcd((*c)->size, p);
            if (div < (*c)->size){
                steps = p % (*c)->size;
                new_cycle_size = (*c)->size / div;
                initial_node = (*c)->nodes[0];
                for (unsigned int j = 0; j < div; j++){
                    current_node = initial_node;
                    initial_node = initial_node->next;
                    new_cycle.reserve(new_cycle_size);
                    for (unsigned int i = 0; i < new_cycle_size; i++){
                        new_cycle.push_back(current_node->index);
                        for (unsigned int k = 0; k < steps; k++){
                            current_node = current_node->next;}}
                    result->cycles.push_back(new Cycle(new_cycle));
                    new_cycle.clear();}}}}
    else if (power < 0){
        unsigned int p = -power;
        unsigned int div;
        std::vector<unsigned int> new_cycle;
        unsigned int new_cycle_size, steps;
        Node *initial_node, *current_node;
        for (std::vector<Cycle*>::iterator c = this->cycles.begin(); c != this->cycles.end(); c++){
            div = __gcd((*c)->size, p);
            if (div < (*c)->size){
                initial_node = (*c)->nodes[0];
                for (unsigned int j = 0; j < div; j++){
                    current_node = initial_node;
                    initial_node = initial_node->previous;
                    new_cycle_size = (*c)->size / div;
                    steps = p % (*c)->size;
                    new_cycle.reserve(new_cycle_size);
                    for (unsigned int i = 0; i < new_cycle_size; i++){
                        new_cycle.push_back(current_node->index);
                        for (unsigned int k = 0; k < steps; k++){
                            current_node = current_node->previous;}}
                    result->cycles.push_back(new Cycle(new_cycle));
                    new_cycle.clear();}}}}
    result->size = result->cycles.size();
    result->minimum --;
    for (std::vector<Cycle*>::iterator c = result->cycles.begin(); c != result->cycles.end(); c++){
        result->maximum = (result->maximum < (*c)->maximum) ? (*c)->maximum : result->maximum;
        result->minimum = (result->minimum < (*c)->minimum) ? result->minimum : (*c)->minimum;}
    return result;}
