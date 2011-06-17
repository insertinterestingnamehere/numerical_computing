import scipy as sp
import scipy.linalg as la
import numpy as np
'''
U,s,Vh = la.svd(A)
S = sp.diag(s)
S = S*(S>tol)
r = sp.count_nonzero(S)
B = sp.dot(U,sp.sqrt(S))
C = sp.dot(sp.sqrt(S),Vh)
B = B[:,0:r]
C = C[0:r,:]
'''

#Problem 3
#When I feed the second example matrix into my function, it comes out with "almost" the correct Drazin Inverse, but the top two rows are 1/2 of what they should be. The other matricies come out right
def drazin(A,tol):
	CB = A.copy()
	
	Bs = []
	Cs = []
	k = 1
	
	while( not (sp.absolute(CB)<tol).all() and sp.absolute(la.det(CB)) < tol):
		U,s,Vh = la.svd(CB)
		S = sp.diag(s)
		S = S*(S>tol)
		r = sp.count_nonzero(S)
		B = sp.dot(U,sp.sqrt(S))
		C = sp.dot(sp.sqrt(S),Vh)
		B = B[:,0:r]
		Bs.append(B)
		C = C[0:r,:]
		Cs.append(C)	
		CB = sp.dot(C,B)
		k+=1
	
	D = sp.eye(A.shape[0])
	for B in Bs:
		D = sp.dot(D,B)
	if( (sp.absolute(CB)<tol).all() ):
		D = sp.dot( D,CB)
	else:
		D = sp.dot( D,np.linalg.matrix_power(CB,-(k+1)))
	for C in reversed(Cs):
		D = sp.dot(D,C)
	return D
