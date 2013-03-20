import numpy as np
import mytimer as mt

def dotijk(A, B):
    rowsA = len(A)
    colsB = len(B[0])
    rowsB = len(B)
    
    result = [[0]*colsB for i in range(rowsA)]
    
    for i in xrange(rowsA):
        for j in xrange(colsB):
            res = 0
            for k in xrange(rowsB):
                res += A[i][k] * B[k][j]
            result[i][j] = res
    return result

def dotikj(A, B):
    rowsA = len(A)
    colsB = len(B[0])
    rowsB = len(B)
    
    result = [[0]*colsB for i in xrange(rowsA)]
    
    for i in xrange(rowsA):
        for j in xrange(rowsB):
            res = 0
            for k in xrange(colsB):
                res += A[i][k] * B[k][j]
            result[i][j] = res
    return result
    
def dot_np(A, B):
    return A.dot(B)
    
def square_arrays():
    sizes = [10, 20, 40, 80]
    with mt.timer() as timer:
        for k in sizes:
            a = [range(i, i+k) for i in range(0, k**2, k)]
            aa = np.array(a)
            timer.time(dotikj, a, a)
            timer.time(dot_np, aa, aa)
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
