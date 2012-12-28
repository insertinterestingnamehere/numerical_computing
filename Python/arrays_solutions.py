import numpy as np
import mytimer as mt

def square(l):
    rows = len(l)
    cols = len(l[0])
    squared = [[0]*cols for i in xrange(rows)]
    
    for i in xrange(rows):
        for j in xrange(cols):
            res = 0
            for k in xrange(cols):
                res += l[i][k] * l[k][j]
            squared[i][j] = res

    return squared

def square_np(l):
    return l.dot(l)

def square_arrays():
    sizes = [10, 20, 40, 80]
    with mt.timer() as timer:
        for k in sizes:
            a = [range(i, i+k) for i in range(0, k**2, k)]
            timer.time(square, a)
            timer.time(square_np, np.array(a))
    return timer.results



def broadcast_1():
    """All input arrays have exactly the same shape"""
    a = np.random.rand(4, 5)
    b = np.random.rand(4, 5)
    r = a * b
    print "Case 1: {} * {} = {}".format(a.shape, b.shape, r.shape)

def broadcast_2():
    """All input arrays are of the same dimension and
    the length of corresponding dimensions match or is 1"""

    a = np.random.rand(5, 4, 1, 6)
    b = np.random.rand(5, 4, 1, 1)
    print "Case 2: {} * {} = {}".format(a.shape, b.shape, r.shape)

def broadcast_3():
    """All input arrays of fewer dimension can have 1
    prepended to their shapes to satisfy the second criteria."""

    a = np.random.rand(1, 6)
    b = np.random.rand(5, 4, 1, 6)
    print "Case 3: {} * {} = {}".format(a.shape, b.shape, r.shape)