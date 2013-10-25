'''
    Solutions file for the MarkovGraph lab.
'''

import numpy as np
from numpy import linalg as la

def Problem1():
    '''
    Compute the answers to problem 1 in the lab.
    Initialize the given Markov matrix, and perform all
    calculations within the function.
    Returns:
        n1 -- n1 is the answer to part 1
        n2 -- n2 is the answer to part 2
    '''
    A = np.array([[.75, .5], [.25, .5]])
    n1 = A.dot(A)[0,0]
    n2 = la.matrix_power(A, 20)[0,0]
    return n1,n2

def Problem2():
    '''
    Calculate the answers to problem 2 in the lab.
    Hint: to see if there is a stable fixed point, raise the
    matrix to some high powers (like powers of 15-20), and see
    if the columns are converging to some common array. If
    they are, return the common array, and if not, return None.
    Returns:
        n1 -- transition matrix asked for in part 1
        n2 -- probability of being in state 2 after 2 
              transitions (starting from state 1)
        n3 -- array giving the stable fixed point if exists, or None
    '''
    A = np.array([[1./4, 1./3, 1./2], [1./4, 1./3, 1./3], [1./2, 1./3, 1./6]])
    n2 = A.dot(A)[1,0]
    n3 = la.matrix_power(A, 20)[:,0]
    return A, n2, n3

'''
    Problem 3
    Implement the function declaration below. Return a tuple of tuples ((n1,n2), (n3,n4))
    where (n1,n2) is a tuple indicating a pair of nodes with the greatest number of paths
    of length 5 between them (there may be more than one correct answer),
    and (n3,n4) is the pair of nodes with no path of length 7
    between them.
'''
def Problem3():
    '''
    Calculate the answers to problem 3 in the lab.
    Returns:
        (n1,n1) -- tuple of values giving a pair of nodes with the
                   greatest no. of paths of length 5 b/n them.
        (n3,n4) -- tuple of values giving the pair of nodes with no
                   path of length 7 b/n them
    '''
    A = np.array([[0, 0, 1, 0, 1, 0, 1],
				  [1, 0, 0, 0, 0, 1, 0],
				  [0, 0, 0, 0, 0, 1, 0],
				  [1, 0, 0, 0, 1, 0, 0],
				  [0, 0, 0, 1, 0, 0, 0],
				  [0, 0, 1, 0, 0, 0, 1],
				  [0, 1, 0, 0, 0, 0, 0]], dtype=np.int64)
    A5 = la.matrix_power(A,5)
    # there are two correct answers, so coords1 has 2 tuples
    coords1 = np.where(A5==np.max(A5))
    A7 = la.matrix_power(A,7)
    coords2 = np.where(A7==0)
    return coords1, coords2
    
def Problem4a():
    '''
    Calculate the Laplacian matrix of the indicated graph required
    in problem 4.
    Returns:
        L -- the laplacian matrix for the required graph
    '''
    A = np.array([[0,1,0,0,1,0],
                  [1,0,1,0,1,0],
                  [0,1,0,1,0,0],
                  [0,0,1,0,1,1],
                  [1,1,0,1,0,0],
                  [0,0,0,1,0,0]])
    return Problem4b(A)

def Problem4b(A):
    ''' 
    Calculate the Laplacian matrix of the graph whose adjacency
    matrix is A. In this function, check that the matrix is undirected
    (i.e. it is symmetric), has no self-edges (this means that the
    diagonal entries must have a certain value), and is unweighted
    (this means that all entries are either 0 or 1).
    If A does not have all of these properties, return None.
    Inputs:
        A -- adjacency matrix of graph
    Returns:
        L -- the Laplacian matrix, or None if the adjacency matrix is bad.
    '''
    # check that A is integer and unweighted
    s = A.diagonal().sum()
    if A.dtype != np.int or np.any(A>1) or np.any(A<0):
        return None
    if s > 0 or (A.T != A).sum() > 0:
        return None
    D = np.diagflat(A.sum(axis=1))
    L = D - A
    return L
