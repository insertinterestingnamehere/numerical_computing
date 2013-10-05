import numpy as np
from scipy import linalg as la

def hqr(A):
	"""Finds the QR decomposition of A using Householder reflectors.
	input: 	A, mxn array with m>=n
	output: Q, orthogonal mxm array
	        R, upper triangular mxn array
	        s.t QR = A
	"""
	R = A.copy()
	m, n = R.shape
	Q = np.eye(m, m)
	for k in xrange(n-1):
		v = R[k:,k].copy()
		v[0] += np.sign(v[0]) * la.norm(v)
		v /= la.norm(v)
		v = v.reshape(m-k, 1)
		P = np.eye(m)
		P[k:,k:] -= 2 * v.dot(v.T)
		Q = P.dot(Q)
		R = P.dot(R)
	return Q.T,R
	
def hess(A):
	"""Computes the upper Hessenberg form of A using Householder reflectors.
	input:  A, mxn array
	output: Q, orthogonal mxm array
			H, upper Hessenberg
			s.t. QHQ' = A
	"""
	H = A.copy()
	m, n = H.shape
	Q = np.eye(m, m)
	for k in xrange(n-2):
		v = H[k+1:,k].copy()
		v[0] += np.sign(v[0]) * la.norm(v)
		v /= la.norm(v)
		v = v.reshape(m-k-1, 1)
		P = np.eye(m, m)
		P[k+1:,k+1:] -= 2 * v.dot(v.T)
		Q = P.dot(Q)
        H = P.dot(H).dot(P.T)
	return Q.T, H
	
def gqr(A):
	"""Finds the QR decomposition of A using Givens rotations.
	input: 	A, mxn array with m>=n
	output: Q, orthogonal mxm array
	        R, upper triangular mxn array
	        s.t QR = A
	"""
	def rotate(i, k, B):
	# create the Givens rotation matrix G to zero out the i,k entry of B
		c,s,r = la.solve(B[k,k], B[i,k])
		r = np.sqrt(B[k,k]**2 + B[i,k]**2)
		c = B[k,k] / r
		s = - B[i,k] / r
		G = np.eye(m)
		G[i,i] = c
		G[k,k] = c
		G[k,i] = -s
		G[i,k] = s
		return G
	
	B = A.copy()	
	m, n = B.shape
	G = np.eye(m)
	#cycle through each nonzero subdiagonal element of B, and rotate it to zero
	for k in xrange(n-1):
		for i in xrange(k+1, m):
			if B[i,k] != 0:
				H = rotate(i, k, B)
				B = H.dot(B)
				G = H.dot(G)
	return G.T, B
