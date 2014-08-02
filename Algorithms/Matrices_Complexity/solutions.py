import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg
from scipy import sparse
from scipy.sparse import linalg as sl


def Problem1():
    # the students should have timed the code 4 times.
    # their runtimes should be similar to what is below
    runtimes = [8.95, 36.7, 144, 557]
    inputs = [1000, 2000, 4000, 8000]
    plt.plot(inputs, runtimes, 'go')
    plt.show()
    
    # now calculate the average ratio of successive runtimes
    return ((36.7/8.95)+(144/36.7)+(557.0/144))/3.0

def Problem2(n):
    # this solution imitates the example code given in the lab
    return np.diagflat([-1]*(n-1), -1) + np.diagflat([2]*n, 0) + np.diagflat([-1]*(n-1),1)

def Problem3(n):
    # build the diagonals
    diags = np.array([[-1]*n,[2]*n,[-1]*n])
    
    # create and return the sparse array
    return sparse.spdiags(diags, [-1,0,1], n, n, format='csr')

def Problem4(n, sparse=False):
    b = np.random.rand(n,1)
    if sparse:
        A = Problem3(n)
        return sl.spsolve(A,b)
    else:
        A = Problem2(n)
        return linalg.solve(A,b)

def Problem5(n):
    A = Problem3(n)
    eig = sl.eigs(A.asfptype(), k=1, which="SM")[0].min()
    return eig*(n**2)
    
#Problem 6
A = np.random.rand(500,500)
b = np.random.rand(500)
%timeit A.dot(b)
B = sparse.csc_matrix(A)
%timeit B.dot(b)
