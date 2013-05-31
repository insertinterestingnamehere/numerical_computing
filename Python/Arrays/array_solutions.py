
import numpy as np
from numpy.random import rand
if __name__ == "__main__":
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

if __name__ == "__main__":
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
	
	#problem 2
	print "problem 2"
	A = rand(1000,1000)
	B = np.empty_like(A)
	for i in xrange(100):
		B[:] = rand(1000,1000)
		A[A<B] = B[A<B]
	np.exp(A, out=A)
	print np.average(np.max(A, axis=1))
	
	#problem 3
	print "problem 3"
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
	
	#problem 4
	print "problem 4"

def laplace(U,tol):
	new = U.copy()
	dif = tol
	while tol<= dif:
		new[1:-1,1:-1] = (U[:-2,1:-1] + U[2:,1:-1] + U[1:-1,:-2] + U[1:-1,2:]) / 4.
		dif = np.max(np.absolute(U-new))
		U[:] = new

if __name__ == "__main__":
	n = 100
	tol=.0001
	U=np.ones((n,n))
	U[:,0] = 100
	U[:,-1] = 100
	U[0] = 0
	U[-1] = 0
	laplace(U, tol)
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	X = np.linspace(0,1,n)
	Y = np.linspace(0,1,n)
	X, Y = np.meshgrid(X, Y)
	ax.plot_surface(X, Y, U, rstride=5)
	plt.show()

#problem 5

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

if __name__ == "__main__":
	print "problem 5"
	broadcast_1()
	broadcast_2()
	broadcast_3()
