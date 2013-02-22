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
        self._maxkeys = branches - 1
        self._minkeys = int(ceil(self._maxkeys/2.0))
        
        self.root = Node(leaf=True)
        self.size = 0
        self.height = 1
        
    def _find(self, key, node):
        i = bisect(node.keys, key)

        if node.leaf:
            if i-1 >= 0 and key == node.keys[i-1].key:
                return node, i
            else:
                return node, -1
        else:
            print "Recursing down ", node.pages[i]
            return self._find(key, node.pages[i])

    def find(self, key):
        if self.root is not None:
            return self._find(key, self.root)
    
    def insert(self, key, value):
        def split_child(node, i):
            #print "Splitting child {} of node {}".format(i, node)
            z = Node()

            y = node.pages[i]
            z.leaf = y.leaf

            #move latter half of keys to node z
            if y.leaf:
                minkeys = self._minkeys

                #connect the leaf nodes a doubly linked list
                z._prev = y
                z._next = y._next
                y._next = z
            else:
                minkeys = self._minkeys + 1

                #move latter half of pages to node z
                z.pages[:], y.pages[:] = (y.pages[minkeys:],
                                        y.pages[:minkeys])

            z.keys[:], m_key, y.keys[:] = (y.keys[minkeys:], 
                                            y.keys[self._minkeys], 
                                            y.keys[:self._minkeys])
                

            node.keys.insert(i, m_key)
            node.pages.insert(i, z)


        def insert_nonfull(node, K):
            i = bisect(node.keys, K)
            if node.leaf:
                #just insert if we are in a leaf node
                node.keys.insert(i, K)
            else:
                if len(node.pages[i].keys) == self._maxkeys:
                    split_child(node, i)
                    if K > node.keys[i]:
                        i += 1
                insert_nonfull(node.pages[i], K)

        K = Key(key, value)

        r = self.root
        if len(r.keys) == self._maxkeys:
            s = Node(leaf=False)
            self.root = s
            s.pages.append(r)
            split_child(s, 0)
            insert_nonfull(s, K)
            self.height += 1
        else:
            insert_nonfull(r, K)