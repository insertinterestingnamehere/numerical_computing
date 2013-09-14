import numpy as np
from numpy.random import rand
from math import sqrt, log
from timeit import timeit as ti

def dist1(A, B):
    # Preallocate output array.
    # Start with everything set to 0.
    D = np.zeros((A.shape[0], B.shape[0]))
    # For each row in A
    for i in xrange(A.shape[0]):
        # For each row in B
        for j in xrange(B.shape[0]):
            # For corresponding entries of
            # the rows in A and B
            for k in xrange(A.shape[1]):
                # Add the squared difference.
                D[i,j] += (A[i,k] - B[j,k])**2
            # Take square root after finishing sum.
            D[i,j] = sqrt(D[i,j])
    return D

def dist2(A, B):
    # Preallocate output array.
    # This time we'll be overwriting it,
    # so there's no need to assign it values.
    D = np.empty((A.shape[0], B.shape[0]))
    # For each row in A
    for i in xrange(A.shape[0]):
        # For each row in B
        for j in xrange(B.shape[0]):
            # Take the distance between the rows
            D[i,j] = sqrt(((A[i] - B[j])**2).sum())
    return D

def dist3(A, B):
    # Preallocate output array
    D = np.empty((A.shape[0], B.shape[0]))
    # For each row in A
    for i in xrange(A.shape[0]):
        # Take distance from this row to each row in B.
        D[i] = np.sqrt(((A[i] - B)**2).sum(axis=1))
    return D

def dist4(A, B):
    # Use the fact that
    # (a-b)**2 = a**2 + b**2 - 2*a*b
    # Take squares of A and B.
    # Reshape so broadcasting works later
    A2 = (A**2).sum(axis=1).reshape((-1,1))
    B2 = (B**2).sum(axis=1).reshape((1,-1))
    # Use matrix multiplication to start
    # computing the remaining term.
    D = A.dot(B.T)
    # Multiply in place to avoid
    # allocating temporary arrays.
    D *= -2
    # Broadcast to add in A2 and B2.
    # Perform these operations in place.
    D += A2
    D += B2
    # Take square root without
    # allocating a temporary array.
    np.sqrt(D, out=D)
    return D

def time(sizes):
    t1 = []
    t2 = []
    t3 = []
    t4 = []
    for size in sizes:
        A = rand(size, 10)
        B = rand(size, 10)
        t1.append(ti("dist1(A, B)", setup="from __main__ import dist1, A, B", number=10))
        t2.append(ti("dist2(A, B)", setup="from __main__ import dist2, A, B", number=10))
        t3.append(ti("dist3(A, B)", setup="from __main__ import dist3, A, B", number=10))
        t4.append(ti("dist4(A, B)", setup="from __main__ import dist4, A, B", number=10))
        print size
    t1 = [log(t)/log(10) for t in t1]
    t2 = [log(t)/log(10) for t in t2]
    t3 = [log(t)/log(10) for t in t3]
    t4 = [log(t)/log(10) for t in t4]
    times = np.empty((5, len(sizes)))
    times[0] = sizes
    times[1] = t1
    times[2] = t2
    times[3] = t3
    times[4] = t4
    np.save("times.npy", times)

if __name__ == __main__:
    time([10, 50, 100, 200, 300, 400, 500, 750, 1000])
