def gqr(A):
	"""Finds the QR decomposition of A using Givens rotations.
	input: 	A, mxn array with m>=n
	output: Q, orthogonal mxm array
	        R, upper triangular mxn array
	        s.t QR = A
	"""
	def rotate(i,k,B):
	# create the Givens rotation matrix G to zero out the i,k entry of B
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
	#cycle through each nonzero subdiagonal element of B, and rotate it to zero
	for k in sp.arange(n-1):
		for i in sp.arange(k+1,m):
			if B[i,k] is not 0:
				H = rotate(i,k,B)
				B = sp.dot(H,B)
				G = sp.dot(H,G)
	return G.T, B