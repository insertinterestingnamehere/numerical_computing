from __future__ import division
import scipy as sp
from scipy import linalg as la
from scipy.linalg import hessenberg
import numpy as np
from numpy.linalg import matrix_power

def qr_solver(A, niter=1000, tol=1e-3):
    '''
    Calculate the eigenvalues of a matrix A using QR iteration.
    Inputs:
        A -- real square matrix
        niter -- number of QR iterations to run
        tol -- real number. All real numbers with magnitude below we treat as 0
    Returns:
        eigs -- list of real or complex numbers, the eigenvalues of A.
    Note: this code is far from optimized, but considering that the algorithm itself is
          inherently limited in robustness, any extra optimization short of chaning the
          algorithm seems wasted effort.
    '''
    # first calculate the upper hessenberg form of A
    H = la.hessenberg(A)
    
    # perform the QR iterations
    for i in xrange(niter):
        Q, R = la.qr(H)
        H = R.dot(Q)
        
    # populate the list of eigenvalues
    eigs = []
    i = 0
    while i < H.shape[0]:
        if i == H.shape[0]-1: #if we are at the last diagonal entry, must be eigenvalue
            eigs.append(H[i,i])
            i += 1
        elif abs(H[i+1,i]) < tol: #if subdiagonal zero, diagonal must be e'value 
            eigs.append(H[i,i])
            i += 1
        else: #subdiagonal nonzero, so find complex e'values of 2x2 block
            a, b, c, d = H[i:i+2,i:i+2].flatten()
            T = a + d
            D = a*d - b*c
            eigs.append(T/2 + math.sqrt(abs(T**2/4 - D))*1j)
            eigs.append(T/2 - math.sqrt(abs(T**2/4 - D))*1j)
            i += 2
    return eigs

def eig(A, normal = False, iter = 100):
	'''Finds eigenvalues of an nxn array A. If A is normal, QRalg.eig 
	may also return eigenvectors.
	
	Parameters
	----------
	A :  nxn array
	     May be real or complex
	normal : bool, optional
		     Set to True if A is normal and you want to calculate
		     the eigenvectors.
	iter : positive integer, optional
			
	Returns
	-------
	v : 1xn array of eigenvectors, may be real or complex
	Q : (only returned if normal = True) 
		nxn array whose columns are eigenvectors, s.t. A*Q = Q*diag(v)
		real if A is real, complex if A is complex
	
	For more on the QR algorithm, see Eigenvalue Solvers lab.
	'''
	def getSchurEig(A):
		#Find the eigenvalues of a Schur form matrix. These are the 
		#elements on the main diagonal, except where there's a 2x2 
		#block on the main diagonal. Then we have to find the 
		#eigenvalues of that block.
		D = sp.diag(A).astype(complex)
		#Find all the 2x2 blocks:
		LD = sp.diag(A,-1)
		index = sp.nonzero(abs(LD)>.01)[0] #is this a good tolerance?
		#Find the eigenvalues of those blocks:
		a = 1
		b = -D[index]-D[index+1]
		c = D[index]*D[index+1] - A[index,index+1]*LD[index]
		discr = sp.sqrt(b**2-4*a*c)
		#Fill in vector D with those eigenvalues
		D[index] = (-b + discr)/(2*a)
		D[index+1] = (-b - discr)/(2*a)
		return D

	n,n = A.shape
	I = sp.eye(n)
	A,Q = hessenberg(A,True)
	if normal == False:
		for i in sp.arange(iter):
			s = A[n-1,n-1].copy()
			Qi,R = la.qr(A-s*I)
			A = sp.dot(R,Qi) + s*I
		v = getSchurEig(A)
		return v
	
	elif normal == True:
		for i in sp.arange(iter):
			s = A[n-1,n-1].copy()
			Qi,R = la.qr(A-s*I)
			A = sp.dot(R,Qi) + s*I
			Q = sp.dot(Q,Qi)
		v = sp.diag(A)
		return v,Q
		
		
def markov(transitions, initial_state, niter):
    '''
    You must write this method.
    Given a transition matrix, calculate the state after a number of 
    iterations using Markov chains, and the unique stable fixed point.
    Inputs:
        transitions -- a transition matrix (all positive values)
        initial_state -- vector of the initial state
        niter -- number of iterations
    Returns:
        cur_state -- the current state
        stable_fixed_point -- the stable fixed point of the matrix
    '''
    pass



def markov_freethrow():
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
    n2 = matrix_power(A, 20)[0,0]
    return n1, n2

def markov_stablept():
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
    n3 = matrix_power(A, 20)[:,0]
    return A, n2, n3