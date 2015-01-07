from __future__ import division
import numpy as np



def cheb_loop(N):
	def p(j1):
		if (j1==0 or j1 == N): return 2.
		else: return 1.
	
	x = np.cos(np.pi*np.arange(N+1)/N)
	D = np.zeros((N+1,N+1))
	# j represents column index
	for j in range(0,N+1):
		for i in range(0,j)+range(j+1,N+1):
			D[i,j] = ((-1.)**(i+j))*p(i)/( p(j)*(x[i]- x[j]) )
	
	
	# Values on the main diagonal
	for j in xrange(1,N): 
		D[j,j] = -x[j]/(2.*(1-x[j]**2.))
	
	D[0,0] = (1.+2.*N**2.)/6.
	D[N,N] = -(1.+2.*N**2.)/6.
	
	return D,x


def cheb_vectorized(N):
	x =  np.cos((np.pi/N)*np.linspace(0,N,N+1))
	x.shape = (N+1,1)
	lin = np.linspace(0,N,N+1)
	lin.shape = (N+1,1)
	
	c = np.ones((N+1,1))
	c[0], c[-1] = 2., 2.
	c = c*(-1.)**lin
	X = x*np.ones(N+1) # broadcast along 2nd dimension (columns)
	
	dX = X - X.T
	
	D = (c*(1./c).T)/(dX + np.eye(N+1))
	D  = D - np.diag(np.sum(D.T,axis=0))
	x.shape = (N+1,)
	# Here we return the differentation matrix and the Chebychev points, 
	# numbered from x_0 = 1 to x_N = -1
	return D, x