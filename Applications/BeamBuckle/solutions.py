import numpy as np
from scipy import linalg as la
import scipy.sparse as spar
from scipy.sparse import linalg as sparla

def problem1(l):
    '''
    print the answer to problem 1 in the Beam buckling lab.
    Inputs:
        l -- length of the beam in feet
    '''
    # initialize constants, do unit conversion
    r = 1.0
    E = 4.35*r*12**2*10**6
    L = l
    I = np.pi*r**4/4
    n = 1000
    h = L/n
    
    # build the tri-diagonal matrix
    diag = -2*np.ones(n)*E*I/h**2
    odiag = np.ones(n-1)*E*I/h**2
    band = np.zeros((2,n))
    band[0,1:] = -odiag
    band[1,:] = -diag
    
    # calculate and print smallest eigenvalue
    evals = la.eigvals_banded(band)
    print evals[0]
    
    # print the analytically calculated answer
    print np.pi**2*E*I/L**2
    
def problem2(l):
    '''
    print the solution to the second problem in Beam Buckling
    Inputs:
        l -- length of beam in feet
    '''
    # initialize constants, unit conversions
    r = 1.0
    E1 = r*12**2*10**7
    E2 = 4.35*r*12**2*10**6
    E3 = 5*r*12**2*10**5
    L = 20.0
    I = np.pi*r**4/4
    n = 100
    h = L/n
    
    # build the sparse matrix B
    b_diag = np.ones(n)
    b_diag[0:n/3] = E1*I/h**2
    b_diag[n/3:n/3+n/3] = E2*I/h**2
    b_diag[n/3+n/3:] = E3*I/h**2
    B = spar.spdiags(b_diag, np.array([0]), n, n, format='csc')

    # build the sparse matrix A
    diag = -2*np.ones(n)
    odiag = np.ones(n)
    A = spar.spdiags(np.vstack((-odiag, -odiag, -diag)),
                     np.array([-1,1,0]), n, n, format='csc')
    
    # calculate and print the smallest eigenvalue                 
    evals = sparla.eigs(B.dot(A), which='SM')
    print evals[0].min()
