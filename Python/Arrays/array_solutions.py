import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from timeit import timeit as ti

#problem 1
def arrmul(A,B):
    new = []
    for i in range(len(A)):
        newrow = []
        for k in range(len(B[0])):
            tot = 0
            for j in range(len(B)):
                tot += A[i][j] * B[j][k]
            newrow.append(tot)
        new.append(newrow)
    return new

def array_vs_list():
    k = 200
    A = [range(i, i+k) for i in range(0, k**2, k)]
    number = 5
    print "problem 1"
    tm = ti("arrmul(A,A)", setup="from __main__ import A, arrmul", number=number)
    print "arrmul(A,A) executed ", number, " times in ", tm, " seconds."
    A = np.array(A)
    number = 5
    tm = ti("np.dot(A,A)", setup="import numpy as np; from __main__ import A", number=number)
    print "np.dot(A,A) executed ", number, " times in ", tm, " seconds."

def fancy_indexing():
	# no longer in lab
    A = rand(1000,1000)
    B = np.empty_like(A)
    for i in xrange(100):
            B[:] = rand(1000,1000)
            A[A<B] = B[A<B]
    np.exp(A, out=A)
    print np.average(np.max(A, axis=1))
    
def time_copy_vs_view():
    A = rand(1000,1000)
    number = 100
    tm = ti("A.reshape(A.size)", setup="from __main__ import A", number=number)
    print "A.reshape(A.size) executed ", number, " times in ", tm, " seconds."
    number = 100
    tm = ti("A.flatten()", setup="from __main__ import A", number=number)
    print "A.flatten() executed ", number, " times in ", tm, " seconds."
    number = ti("A.reshape((1,A.size))", setup="from __main__ import A", number=number)
    print "A.reshape((1,A.size)) executed ", number, " times in ", tm, " seconds."
    
    #part 2 of problem 3
    print "the difference is that np.vstack(A) returns a new array"
    print "while A.T returns a view"
    print "A.T is much faster"
    A = rand(1,1000000)
    number = 500
    tm = ti("np.vstack(A)", setup="import numpy as np; from __main__ import A", number=number)
    print "np.vstack(A) executed ", number, " times in ", tm, " seconds."
    number = 500
    tm = ti("A.T", setup="from __main__ import A", number=number)
    print "A.T executed ", number, " times in ", tm, " seconds."
    
def laplace(U, tol):
    new = U.copy()
    dif = tol
    while tol <= dif:
        new[1:-1,1:-1] = (U[:-2,1:-1] + U[2:,1:-1] + U[1:-1,:-2] + U[1:-1,2:]) / 4.
        dif = np.max(np.absolute(U-new))
        U[:] = new

def large_numbers(n):
	# demonstrates law of large numbers
	# as n increases, variance goes to 0.
    A = rand(n, n)
    return A.mean(axis=1).var()

# The problems from here on are no longer in the first lab.
def prob5():
    im = np.random.randint(1,256,(100,100,3))
    b = np.array([0.5,0.5,1])
    im_bluer = (im * b).astype(int)

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
    r = a * b
    print "Case 2: {} * {} = {}".format(a.shape, b.shape, r.shape)

def broadcast_3():
    """All input arrays of fewer dimension can have 1
    prepended to their shapes to satisfy the second criteria."""

    a = np.random.rand(1, 6)
    b = np.random.rand(5, 4, 1, 6)
    r = a * b
    print "Case 3: {} * {} = {}".format(a.shape, b.shape, r.shape)

