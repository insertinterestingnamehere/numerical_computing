import numpy as np
from scipy import linalg as la
from math import copysign

def hqr(A):
    """Finds the QR decomposition of A using Householder reflectors.
    input: 	A, mxn array with m>=n
    output: Q, orthogonal mxm array
            R, upper triangular mxn array
            s.t QR = A
    """
    # This is just a pure Python implementation.
    # It's not fully optimized, but it should
    # have the right asymptotic speed.
    # initialize Q and R
    # start Q as an identity
    # start R as a C-contiguous copy of A
    # take a transpose of Q to start out
    # so it is C-contiguous when we return the answer
    Q = np.eye(A.shape[0]).T
    R = np.array(A, order="C")
    # initialize m and n for convenience
    m, n = R.shape
    # avoid reallocating v in the for loop
    v = np.empty(A.shape[1])
    for k in xrange(n-1):
        # get a slice of the temporary array
        vk = v[k:]
        # fill it with corresponding values from R
        vk[:] = R[k:,k]
        # add in the term that makes the reflection work
        vk[0] += copysign(la.norm(vk), vk[0])
        # normalize it so it's an orthogonal transform
        vk /= la.norm(vk)
        # apply projection to R
        R[k:,k:] -= 2 * np.outer(vk, vk.dot(R[k:,k:]))
        # Apply it to Q
        Q[k:,:] -= 2 * np.outer(vk, vk.dot(Q[k:,:]))
    # note that its returning Q.T, not Q itself
    return Q.T, R

def hess(A):
	"""Computes the upper Hessenberg form of A using Householder reflectors.
	input:  A, mxn array
	output: Q, orthogonal mxm array
			H, upper Hessenberg
			s.t. QHQ' = A
	"""
	H = A.copy()
	m, n = H.shape
	Q = np.eye(m)
	for k in xrange(n-2):
		v = H[k+1:,k].copy()
		v[0] += np.sign(v[0]) * la.norm(v)
		v /= la.norm(v)
		v = v.reshape(m-k-1, 1)
		P = np.eye(m)
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
