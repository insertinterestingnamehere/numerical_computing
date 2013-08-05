from collections import deque

class Node(object):
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
        self.height = 1
    
    def free(self):
        self.data = None
        self.left = None
        self.right = None

    def __str__(self):
        return str(self.data)
       
class BinTree(object):
    def __init__(self):
        self.root = None
        self.size = 0
        
    def clear(self):
        """Recursively clear the AVL Tree"""
        def _clear(self, n):
            if n is not None:
                _clear(n.left)
                _clear(n.right)
                n.free()
                
        _clear(self.root)
        self.root = None
        self.size = 0
        
    def _set_height(self, n):
        if n is None:
            return 0
        else:
            return 1 + max(getattr(n.left, "height", 0), getattr(n.right, "height", 0))
        
    def insert(self, item):
        
        def _recur_insert(n, cand):
            if n is None:
                return Node(cand)
            else:
                if cand < n.data:
                    n.left = _recur_insert(n.left, cand)
                elif cand > n.data:
                    n.right = _recur_insert(n.right, cand)
                else:
                    return n
            
            n.height = self._set_height(n)
            return n
    
        if self.root is None:
            self.root = Node(item)
        else:
            self.root = _recur_insert(self.root, item)
    
        self.root.height = self._set_height(self.root)
        self.size += 1
        
    def remove(self, item):
        
        def _recur_remove(n, cand):
            if n is None:
                return
            else:
                if cand < n.data:
                    n.left = _recur_remove(n.left, cand)
                elif cand > n.data:
                    n.right = _recur_remove(n.right, cand)
                elif cand == n.data:
                    if n.left is None and n.right is None:
                        return
                    elif n.left is not None and n.right is None:
                        nleft = n.left
                        del n
                        return nleft
                    elif n.left is None and n.right is not None:
                        nright = n.right
                        del n
                        return nright
                    else:
                        nmin = n.right
                        while nmin.left is not None:
                            nmin = nmin.left
                        
                        n.data, nmin.data = nmin.data, n.data
                        
                        n.right = _recur_remove(n.right, nmin.data)
                        return n
            
                if n is not None:
                    n.height = self._set_height(n)
                
                return n
            
        
        if self.root is None:
            return
        else:
            self.root = _recur_remove(self.root, item)
            
        if self.root is not None:
            self.root.height = self._set_height(self.root)
            
        self.size -= 1
    
    def find(self, item):
        n = self.root
        while n is not None:
            if item < n.data:
                n = n.left
            elif item > n.data:
                n = n.right
            else:
                return n
    
 
def print_tree(tree_root):
    parents = deque()
    children = deque()
    parents.append(tree_root)

    level = 1
    while len(parents) > 0 or len(children) > 0:
        print "Level {}: {}".format(level,
                                    ', '.join(str(n) for n in parents))
        while len(parents) > 0:
            node = parents.popleft()
            

            if node.left is not None:
                children.append(node.left)
            if node.right is not None:
                children.append(node.right)

        parents, children = children, parents
        level += 1
    
