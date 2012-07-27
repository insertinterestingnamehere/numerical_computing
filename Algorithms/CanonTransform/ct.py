from __future__ import division
import scipy as sp
import scipy.linalg as la



def hqr(A):
	""" Using Householder rotations, computes the QR decomposition for
	an mxn matrix A, where m>=n.
	input: A -- array, mxn
	   getQ -- Boolean, tells whether to calculate Q (default True)
	output: 
		Q -- orthonormal mxm array
		R -- upper triangular mxn array, s.t. QR = A"""
	R = A.copy()
	m,n = R.shape
	Q = sp.eye(m,m)
	for k in sp.arange(n-1):
		v = R[k:m,k].copy() 
		v[0] += sp.sign(v[0])*la.norm(v)
		v = v/la.norm(v)
		v = v.reshape(m-k,1)
		P = sp.eye(m,m)
		P[k:m,k:m] -= 2*sp.dot(v.copy(),v.copy().T)
		Q = sp.dot(P,Q)
		R = sp.dot(P,R)
	return Q.T,R
	
def hess(A):
	R = A.copy()
	m,n = R.shape
	Q = sp.eye(m,m)
	for k in sp.arange(n-2):
		v = R[k+1:m,k].copy()
		v[0] += sp.sign(v[0])*la.norm(v)
		v = v/la.norm(v)
		v = v.reshape(m-k-1,1)
		P = sp.eye(m,m)
		P[k+1:m,k+1:m] -= 2*sp.dot(v.copy(),v.copy().T)
		Q = sp.dot(P,Q)
		R = sp.dot(P,sp.dot(R,P.T))
	return Q.T,R
	
def gqr(A):
	"""Finds the QR decomposition of A using Givens rotations.
	input: 	A, mxn array with m>=n
	output: Q, orthogonal mxm array
	        R, upper triangular mxn array
	        s.t QR = A
	"""
	def rotate(i,k,B):
	# create the Givens rotation matrix G to zero out the 
	# i,k entry of B
		c,s,r = solve(B[k,k],B[i,k])
		r = sp.sqrt(B[k,k]**2 + B[i,k]**2)
		c = B[k,k]/r
		s = -B[i,k]/r
		G = sp.eye(m)
		G[i,i] = c
		G[k,k] = c
		G[k,i] = -s
		G[i,k] = s
		return G
	
	B = A.copy()	
	m,n = B.shape
	G = sp.eye(m)
	#cycle through each subdiagonal element of B, and zero it out
	for k in sp.arange(n-1):
		for i in sp.arange(k+1,m):
			if B[i,k] is not 0:
				H = rotate(i,k,B)
				B = sp.dot(H,B)
				G = sp.dot(H,G)
	return G.T, B