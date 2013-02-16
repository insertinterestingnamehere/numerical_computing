from math import ceil
from bisect import bisect
from itertools import islice, cycle

class Key(object):
    """Key object for B+Tree

    Keys in noninternal nodes contain a node pointer for their data field.
    Keys in leaf nodes contain pointer to their actual data
    """

    def __init__(self, key, data=None):
        self.key = key
        self.data = key
        
    def __eq__(self, other):
        return self.key == other.key
    
    def __lt__(self, other):
        return self.key < other.key
    
    def __repr__(self):
        return "<{}; {}>".format(self.key.__repr__(), self.data.__repr__())
    
class Node(object):
    """
    self.leaf = True if leaf Node, else index node
    self.keys
    """
    def __init__(self, leaf=True, n=None, prev=None, parent=None):
        self.leaf = leaf
        self.pages = []
        self.keys = []
        self._next = n
        self._prev = prev
        self._parent = parent
        self.nkeys = 0
    
    def __repr__(self):
        return "{}: {}".format('Leaf' if self.leaf else 'Internal', self.keys)

    def insert(self, key):
        i = binsearch(self.keys, key)

        #check the see if key already exists
        if i:
            self.pages[i] = key
        else:
            i = bisect(self.pages, key)
            self.pages.insert(i, key)
            self.nkeys += 1

class BTree(object):
    def __init__(self, branches):
        self._minkeys = int(ceil(branches/2.0))
        self._maxkeys = branches
        self.root = None
        self.size = 0
        
    def _find(self, key, node):
        i = bisect(node.keys, key)

        if node.leaf:
            if i-1 >= 0 and key == node.keys[i-1]:
                return node, i-1
            else:
                return node, -1
        else:
            return self._find(key, node.pages[i])

    def find(self, key):
        if self.root is not None:
            if key is not Key:
                key = Key(key)
            return self._find(key, self.root)
    
    def insert(self, key, value):
        def split_node(node):
            if not node.leaf:
                new_node = Node(parent=node._parent)
            else:
                new_node = Node(n=node._next, prev=node, parent=node._parent)

            
            #move last half of old leaf to new leaf
            nkeys = int(((node.nkeys - 1)/2.) + 1)
            for i in xrange(nkeys):
                new_node.insert(node.pages.pop())

            if node.leaf:
                #copy up smallest key into parent
                copykey = Key(node[0], node)
                node._parent.insert(copykey)
            else:
                #internal node
                #push up smallest key into parent
                node._parent.insert(node.pages.pop(0))

            if node._parent is None:
                #we are at the root and need to split
                newroot = Node()
                newroot.keys.append(node.pages.pop(0))
                newroot.pages.extend([node, new_node])
                self.root = newroot


            return new_node

        key = Key(key, value)
        leaf_node = self.search(key)

        #insert the new key
        leaf_node.insert(key)

        #check to see if node overflowed
        _inode = leaf_node
        while len(_inode) > self._maxkeys:
            n = split_node(_inode)
            _inode = n._parent


    def remove(self, key):
        def join_leaf(self, leaf):
            pass
        pass