#pragma once

class Node{
    public:
        unsigned int index;
        Node* previous;
        Node* next;
        Node(unsigned int index, Node* previous, Node* next);
        bool operator<(Node other);
        bool operator>(Node other);
        bool operator<=(Node other);
        bool operator>=(Node other);
        bool operator==(Node other);
        bool operator!=(Node other);};
