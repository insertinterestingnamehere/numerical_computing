#include "Cycle.hpp"
#include <vector>
#include <algorithm>

// This one is private.
// It is a constructor that does not verify ordering or consistency.
Cycle::Cycle(std::vector<Node*>& nodearr, unsigned int size, unsigned int minimum, unsigned int maximum){
    // Set size and array immediately.
    // Sorting will be performed outside this
    // function whenever necessary.
    this->size = size;
    this->nodes = nodearr;
    this->minimum = minimum;
    this->maximum = maximum;}

// Copy Constructor
Cycle::Cycle(const Cycle& other){
    this->size = other.size;
    this->minimum = other.minimum;
    this->maximum = other.maximum;
    this->nodes.reserve(this->size);
    for (unsigned int i = 0; i < other.nodes.size(); i++){
        // Copy all the nodes.
        this->nodes.push_back(new Node(*(other.nodes[i])));}}

// Assignment operator
Cycle& Cycle::operator=(const Cycle& other){
    this->minimum = other.minimum;
    this->maximum = other.maximum;
    // Delete old nodes
    for (unsigned int i = 0; i < this->nodes.size(); i++){
        delete this->nodes[i];}
    this->size = other.size;
    this->nodes.reserve(this->size);
    for (unsigned int i = 0; i < other.nodes.size(); i++){
        this->nodes.push_back(new Node(*(other.nodes[i])));}}
    

// Another private constructor to be used by the Permutation class.
Cycle::Cycle(std::list<unsigned int>& indexlist, unsigned int size){
    // Set size attribute and find out how much memory to allocate.
    this->size = size;
    // Reserves space for the node pointers.
    (this->nodes).reserve(this->size);
    // Initialize the maximum and the minimum
    // so they can be updated as we construct the nodes.
    this->maximum = indexlist.front();
    this->minimum = indexlist.front();
    std::list<unsigned int>::iterator it = indexlist.begin();
    // Start constructing the nodes.
    // Do the first one with the first index
    // and two null pointers.
    nodes.push_back(new Node(*it, 0, 0));
    it++;
    // Now iterate through the rest of the nodes.
    for (unsigned int i=1; i<size; i++){
        // Allocate a new node with an appropriately set previous pointer.
        // The next item hasn't been set yet, so just set it to null for now.
        nodes.push_back(new Node(*it, nodes[i-1], 0));
        // Make the next pointer for the previous node point to the
        // node we just created.
        nodes[i-1]->next = nodes[i];
        // Update the maximum and the minimum.
        // Check Maximum first since that will change most often.
        // Only check minimum if maximum test fails.
        if (maximum<(*it)){this->maximum = *it;}
        else if ((*it)<minimum){this->minimum = *it;}
        it++;}
    // Now we connect the first and last node.
    // Set the next pointer of the last node to point to the first one.
    nodes[size-1]->next = nodes[0];
    // Set the previous pointer of the first node to point to the last one.
    nodes[0]->previous = nodes[size-1];
    // Sort nodes by their indices.
    // This is necessary because several of the other methods
    // require that the nodes be sorted by index.
    this->sort();}

// Various other constructors:
// First with a list.
Cycle::Cycle(std::list<unsigned int>& indexlist){
    // Set size attribute and find out how much memory to allocate.
    this->size = indexlist.size();
    // Reserve space for the node pointers.
    (this->nodes).reserve(this->size);
    // Start constructing the Nodes themselves.
    // Do the first one with the first index
    // and two null pointers.
    this->nodes.push_back(new Node(indexlist.front(), 0, 0));
    // Initialize the maximum and the minimum
    // so they can be updated as we construct the nodes.
    this->maximum = indexlist.front();
    this->minimum = indexlist.front();
    // Iterate over the remaining indices that need to
    // have nodes created for them.
    std::list<unsigned int>::iterator it = indexlist.begin();
    it++;
    for (unsigned int i=1; i<(this->size); i++){
        // Allocate a new node with an appropriately set previous pointer.
        // The next item hasn't been set yet, so just set it to null for now.
        this->nodes.push_back(new Node(*it, nodes[i-1], 0));
        // Make the next pointer for the previous node point to the
        // node we just created.
        nodes[i-1]->next = nodes[i];
        // Update the maximum and the minimum.
        // Check Maximum first since that will change most often.
        // Only check minimum if maximum test fails.
        if (maximum<(*it)){this->maximum = *it;}
        else if ((*it)<minimum){this->minimum = *it;}
        it++;}
    // Now we connect the first and last node.
    // Set the next pointer of the last node to point to the first one.
    nodes[size-1]->next = nodes[0];
    // Set the previous pointer of the first node to point to the last one.
    nodes[0]->previous = nodes[size-1];
    // Sort nodes by their indices.
    // This is necessary because several of the other methods
    // require that the nodes be sorted by index.
    this->sort();}

Cycle::Cycle(std::vector<unsigned int>& indexvector){
    // Set size attribute and find out how much memory to allocate.
    this->size = indexvector.size();
    // Reserve space for the Node pointers.
    (this->nodes).reserve(this->size);
    // Start constructing the Nodes themselves.
    // Do the first one with the first index
    // and two null pointers.
    this->nodes.push_back(new Node(indexvector[0], 0, 0));
    // Initialize the maximum and the minimum
    // so they can be updated as we construct the nodes.
    this->maximum = indexvector[0];
    this->minimum = indexvector[0];
    // Iterate over the remaining indices that need to
    // have nodes created for them.
    for (unsigned int i=1; i<(this->size); i++){
        // Allocate a new node with an appropriately set previous pointer.
        // The next item hasn't been set yet, so just set it to null for now.
        this->nodes.push_back(new Node(indexvector[i], nodes[i-1], 0));
        // Make the next pointer for the previous node point to the
        // node we just created.
        nodes[i-1]->next = nodes[i];
        // Update the maximum and the minimum.
        // Check Maximum first since that will change most often.
        // Only check minimum if maximum test fails.
        if (maximum<indexvector[i]){this->maximum = indexvector[i];}
        else if (indexvector[i]<minimum){this->minimum = indexvector[i];}}
    // Now we connect the first and last node.
    // Set the next pointer of the last node to point to the first one.
    nodes[size-1]->next = nodes[0];
    // Set the previous pointer of the first node to point to the last one.
    nodes[0]->previous = nodes[size-1];
    // Sort nodes by their indices.
    // This is necessary because several of the other methods
    // require that the nodes be sorted by index.
    this->sort();}

/*
Cycle::Cycle(unsigned int indexarr, unsigned int size, bool sorted){
    // Set size attribute.
    this->size = size;
    // Allocate the array of Node pointers.
    this->nodes = new std::vector<Node*>;
    (this->nodes).reserve(this->size);
    // Start constructing the Nodes themselves.
    // Do the first one with the first index
    // and two null pointers.
    nodes[0] = new Node(indexarr[0], 0, 0);
    // Iterate over the remaining indices that need to
    // have nodes created for them.
    for (unsigned int i=1; i<size; i++){
        // Allocate a new node with an appropriately set previous pointer.
        // The next item hasn't been set yet, so just set it to null for now.
        nodes[i] = new Node(indexarr[i], nodes[i-1], 0);
        // Make the next pointer for the previous node point to the
        // node we just created.
        nodes[i-1]->next = nodes[i];
        // Update the maximum and the minimum.
        // Check Maximum first since that will change most often.
        // Only check minimum if maximum test fails.
        if (maximumM(*it)){this->maximum = *it;}
        else if ((*it)<minimum){this->minimum = *it;}}
    // Now we connect the first and last node.
    // Set the next pointer of the last node to point to the first one.
    nodes[size-1]->next = nodes[0];
    // Set the previous pointer of the first node to point to the last one.
    nodes[0]->previous = nodes[size-1];
    // Sort nodes by their indices.
    // This is necessary because several of the other methods
    // require that the nodes be sorted by index.
    this->sort();}
*/

Cycle::~Cycle(){
    // Deallocate each node.
    // The vector of pointers will be taken care of automatically.
    for (unsigned int i=1; i<(this->size); i++){
        delete nodes[i];}}

// This next function is not a part of the class, but it
// is necessary for sorting using the standard library.
//bool __node_compare(Node* first, Node* second){
//    // Essentially, compare two pointers to nodes by
//    // comparing their indices.
//    return (first->index)<(second->index);}

// [](Node* i, Node* j){return (i->index) < (j->index);}
struct __node_compare{
  bool operator() (Node* i, Node* j) {return (i->index)<(j->index);}};

void Cycle::sort(){
    // Sort using std::sort and the comparison function we just defined.
    std::sort(this->nodes.begin(), this->nodes.end(), __node_compare());}

bool Cycle::verify(){
    // Since the Cycle is sorted we can check for adjacent matches.
    // This ensures that it actually is a valid cycle.
    for (unsigned int i=1; i<(this->size); i++){
        if ((this->nodes)[i-1]->index == (this->nodes)[i]->index){
            return false;}}
    return true;}

std::string Cycle::get_string(){
    // Not super efficient, but it should be fine for anything
    // someone would actually want to represent as a string.
    // Use a stringstream object to control input and output.
    std::stringstream stream;
    // Start with a parenthesis.
    stream << "(";
    // Now add in the index for each node.
    // We'll iterate using this pointer:
    Node* current = (this->nodes)[0];
    // Iterate through the nodes in the order of permutation.
    while (current!=((this->nodes)[0]->previous)){
        // Write the current index to the stringstream.
        // Separate indices by the character "~".
        stream << current->index << "~";
        // Update the pointer.
        current = current->next;}
    // Write the last node's index to the stringstream.
    stream << current->index << ")";
    // Write the stringstream to a string.
    std::string str;
    stream >> str;
    // Return it.
    return str;}

// Allow for viewing of attributes.
unsigned int Cycle::get_min(){
    return this->minimum;}

unsigned int Cycle::get_max(){
    return this->maximum;}

unsigned int Cycle::get_size(){
    return this->size;}

bool Cycle::in(unsigned int index){
    // Test for membership using a binary search.
    // An index is in a cycle if the cycle does not fix that index.
    // Assumes nodes are sorted by index.
    // Use std::binary_search and the __node_compare function we defined earlier.
    // Make a dummy node to use for comparison.
    Node idx_node = Node(index, 0, 0);
    return std::binary_search(this->nodes.begin(), this->nodes.end(), &idx_node, __node_compare());}

unsigned int Cycle::trace(unsigned int index){
    // Use a binary search to trace an index through a permutation.
    // If an item starts at a given index, this method returns the index where
    // it will be located after the Cycle has been applied.
    // Assumes nodes are sorted by index.
    // Use std::lower_bound and the __node_compare function we defined earlier.
    // Find the first node index that is not less than the given index.
    Node idx_node = Node(index, 0, 0);
    std::vector<Node*>::iterator n = std::lower_bound(this->nodes.begin(), this->nodes.end(), &idx_node, __node_compare());
    // If the index is larger than the indices of all the nodes in the cycle just return the original index.
    if (n == this->nodes.end()){return index;}
    // If the index isn't actually in the Cycle we just return the index itself.
    if ((*n)->index != index){return index;}
    // If it is changed by the cycle, we return its image under the cycle.
    return (*n)->next->index;}

Cycle* Cycle::inverse(){
    // Construct an inverse cycle.
    // As long as the original cycle is ordered properly, the
    // inverse will also be ordered properly.
    // Allocate the array of node pointers to be used in the new cycle.
    std::vector<Node*> new_nodes;
    new_nodes.reserve(this->size);
    // Allocate all the new nodes with their next and previous pointers
    // reversed so that it represents the inverse cycle.
    // Use pointers as iterators.
    for (std::vector<Node*>::iterator i = (this->nodes).begin(); i!=(this->nodes).end(); i++){
        // Set the current pointer to point to the newly allocated node.
        // Notice that the next and previous values are reversed.
        new_nodes.push_back(new Node((*i)->index, (*i)->next, (*i)->previous));}
    // Construct a new cycle from the array of nodes we have just made.
    // This will initialize it as a pointer.
    return new Cycle(new_nodes, this->size, this->minimum, this->maximum);}

unsigned int Cycle::trace_inverse(unsigned int index){
    // Use a binary search to trace an index through a permutation.
    // If an item starts at a given index, this method returns the index where
    // it will be located after the Cycle has been applied.
    // Assumes nodes are sorted by index.
    // Use std::lower_bound and the __node_compare function we defined earlier.
    // Find the first node index that is not less than the given index.
    Node idx_node = Node(index, 0, 0);
    std::vector<Node*>::iterator n = std::lower_bound(this->nodes.begin(), this->nodes.end(), &idx_node, __node_compare());
    // If the index isn't actually in the Cycle we just return the index itself.
    if ((*n)->index != index){return index;}
    // Otherwise we return its image under the inverse cycle.
    return (*n)->previous->index;}

// Define comparison operators in case we want them later.
bool Cycle::operator<(const Cycle& other){
    return (this->nodes[0]->index) < (other.nodes[0]->index);}

bool Cycle::operator>(const Cycle& other){
    return (this->nodes[0]->index) > (other.nodes[0]->index);}

bool Cycle::operator<=(const Cycle& other){
    return (this->nodes[0]->index) <= (other.nodes[0]->index);}

bool Cycle::operator>=(const Cycle& other){
    return (this->nodes[0]->index) >= (other.nodes[0]->index);}

bool Cycle::operator==(const Cycle& other){
    return (this->nodes[0]->index) == (other.nodes[0]->index);}

bool Cycle::operator!=(const Cycle& other){
    return (this->nodes[0]->index) != (other.nodes[0]->index);}
