from __future__ import division
import scipy as sp
from scipy import linalg as la
from scipy.linalg import hessenberg

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