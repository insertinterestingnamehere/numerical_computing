#include "Node.hpp"

Node::Node(unsigned int index, Node* previous, Node* next){
    this->index = index;
    this->previous = previous;
    this->next = next;}

bool Node::operator<(Node other){
    return this->index < other.index;}

bool Node::operator>(Node other){
    return this->index > other.index;}

bool Node::operator<=(Node other){
    return this->index <= other.index;}

bool Node::operator>=(Node other){
    return this->index >= other.index;}

bool Node::operator==(Node other){
    return this->index == other.index;}

bool Node::operator!=(Node other){
    return this->index != other.index;}
