
# coding: utf-8

# In[42]:

# Problem 1
class Node():
    def __init__(self, data):
        self.next = None
        self.value = data

class SLinkedList(object):
    def __init__(self):
        self.head = None
        self.tail = None
        self.counter = 0
        
    def size(self):
        return self.counter
    
    def find(self, index):
        nfind = self.head
        nprev = None
        count = 0
        if index >= self.counter:
            raise IndexError
        else:
            while count < index and nfind.next:
                count += 1
                nprev = nfind
                nfind = nfind.next
            return nprev, nfind
    
    def insert(self, index, data):
        n = Node(data)
        if index > self.counter:
            raise IndexError
        if index == 0: 
            n.next = self.head
            self.head = n
            if self.counter == 0:
                self.tail = self.head
        elif index == self.counter: 
            self.tail.next = n
            self.tail = n
        else: 
            nprev, nindex = self.find(index)
            n.next = nindex
            nprev.next = n
        self.counter += 1
    
    def remove(self, index):
        if index > self.counter:
            raise IndexError
        if index == 0: 
            self.head = self.head.next
            if self.counter == 1:
                self.head, self.tail = None
        else: 
            nprev, nindex = self.find(index)
            nprev.next = nindex.next
            if nindex == self.tail:
                self.tail = nprev
        self.counter -= 1

    def clear(self):
        self.head = None
    
    def __str__(self):
        return '[' + ','.join(map(str, iter(self))) + ']'
    
    def __iter__(self):
        temp = self.head
        while temp:
            yield temp.value
            temp = temp.next

# Problem 2
class Stack(SLinkedList):
    def insert(self, data):
        SLinkedList.insert(self, self.counter, data)
    
    def remove(self):
        SLinkedList.remove(self, (self.counter-1))

class Queue(SLinkedList):
    def insert(self, data):
        SLinkedList.insert(self, self.counter, data)
    
    def remove(self):
        SLinkedList.remove(self, 0)


# In[72]:

# Problems 3 and 4:
class Node(object):
    def __init__(self, data):
        self.value = data
        self.left = None
        self.right = None
        
    def __str__(self):
        return str(self.value)
        
class BinTree(object):
    def __init__(self):
        self.root = None
        self.size = 0
    
    def insert(self, data):
        def _recur_insert(node, item):
            if node is None:
                return Node(item)
            else:
                if item < node.value:
                    node.left = _recur_insert(node.left, item)
                elif item > node.value:
                    node.right = _recur_insert(node.right, item)
            return node
        self.root = _recur_insert(self.root, data)
        self.size += 1
    
    def find(self, data):
        temp = self.root
        while temp is not None:
            if data < temp.value:
                temp = temp.left
            elif data > temp.value:
                temp = temp.right
            else:
                return temp
        return False
    
    def remove(self, item):
        
        def _recur_remove(n, cand):
            if n is None:
                return
            else:
                # Travel to the left subtree
                if cand < n.value:
                    n.left = _recur_remove(n.left, cand)
                # Travel to the right subtree
                elif cand > n.value:
                    n.right = _recur_remove(n.right, cand)
                # Found Node
                elif cand == n.value:
                    # No children
                    if n.left is None and n.right is None:
                        return
                    # One left child
                    elif n.left is not None and n.right is None:
                        nleft = n.left
                        del n
                        return nleft
                    # One right child
                    elif n.left is None and n.right is not None:
                        nright = n.right
                        del n
                        return nright
                    # Two children
                    else:
                        nmin = n.right
                        # Identify in-order successor
                        while nmin.left is not None:
                            nmin = nmin.left
                        
                        # Switch the data between node and in-order sucessor
                        n.value, nmin.value = nmin.value, n.value
                        # Continue recursive function, remove n when it is reached (now has either one child or no children)
                        n.right = _recur_remove(n.right, nmin.value)
                        return n 
                return n
            
        
        if self.root is None:
            return
        else:
            self.root = _recur_remove(self.root, item)    
        self.size -= 1


# In[88]:


#Problems 5 and 6
def HashFunction(string, tablesize, prime=5417):
    index = (len(string)*prime) % tablesize
    return index

class HashTable(object):
    def __init__(self, hsize, hashfunc):
        self.hashtable = [None]*hsize
        self.elsize = 0
        self.hsize = hsize
        self._hash = hashfunc    
    def find(self, data):
        index = self._hash(data, self.hsize)
        for i in xrange(self.hsize):
            newindex = (index + i) % self.hsize
            print 'Searching ' + str(newindex)
            if self.hashtable[newindex] == data:
                return True
        return False
    def insert(self, data):
        index = self._hash(data, self.hsize)
        i = 0
        newindex = index
        # Ensures the table isn't full.
        if self.elsize < self.hsize:
            # Identifies the next empty index.
            while self.hashtable[newindex] is not None:
                newindex = (newindex + 1) % self.hsize
            self.hashtable[newindex] = data
            self.elsize += 1
        # Returns an error if the table is full.
        else:
            print 'Sorry, there is no more room in this hash table; please resize before you attempt to insert ' + data + '.'

H = HashTable(11, HashFunction)
Felines = ['lion', 'tiger', 'cheetah', 'cougar', 'colocolo', 'cat']
for i in Felines:
    H.insert(i)
print H.hashtable
H.insert('clouded leopard')
H.insert('jaguar')
print H.hashtable
# Problem 5: Upon inserting the latter two felines, 'jaguar' replaced 'cougar' and 'clouded leopard' replaced 'lion, 
# which was nothing if not a burning shame. However, H.elsize still maintains that 8 elements are contained within
# the hash table which is incorrect. 
# Problem 6: We find 'clouded leopard' at index 10 and 'jaguar' at index 0.

# Problems 7 and 8
class HashTable(object):
    def __init__(self, hsize, hashfunc):
        self.hashtable = [None]*hsize
        self.elsize = 0
        self.hsize = hsize
        self._hash = hashfunc
    def load(self):
        loadfactor = float(self.elsize)/self.hsize
        return loadfactor
    # Problem 8
    def resize(self):
        loadfactor = self.load()
        if loadfactor > .75:
            while loadfactor >= 0.33:
                self.hsize +=1
                loadfactor = self.load()
            newhash = [None]*self.hsize
            for i in self.hashtable:
                if i is not None:
                    for j in i:
                        newindex = self._hash(j, self.hsize)
                        if newhash[newindex] is None:
                            newhash[newindex] = [j]
                        else:
                            newhash[newindex].append(j)
            self.hashtable = newhash            
    # Problem 7
    def insert(self, data):
        index = self._hash(data, self.hsize)
        if self.hashtable[index] is None:
            self.hashtable[index] = [data]
        else:
            self.hashtable[index].append(data)
        self.elsize +=1
        # Problem 8
        self.resize()
    def find(self, data):
        index = self._hash(data, self.hsize)
        for i in self.hashtable[index]:
            if i == data:
                return True
        return False
H = HashTable(11, HashFunction)
Felines = ['lion', 'tiger', 'cheetah', 'cougar', 'colocolo', 'cat', 'clouded leopard', 'jaguar']
for i in Felines:
    H.insert(i)
print H.hashtable
print H.hsize
H.insert('bengal tiger')
H.insert('siberian tiger')
H.insert('liger')
print 'The new hash table size is ' + str(H.hsize)


# In[ ]:



