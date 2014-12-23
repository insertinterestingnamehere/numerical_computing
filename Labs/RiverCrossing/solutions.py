import numpy as np


def cheb(N):
	x =  np.cos((np.pi/N)*np.linspace(0,N,N+1))
	x.shape = (N+1,1)
	lin = np.linspace(0,N,N+1)
	lin.shape = (N+1,1)
	
	c = np.ones((N+1,1))
	c[0], c[-1] = 2., 2.
	c = c*(-1.)**lin
	X = x*np.ones(N+1) # broadcast along 2nd dimension (columns)
	
	dX = X - X.T
	# print " x = \n", x, "\n"
	# print " c = \n", c, "\n"
	# print " X = \n", X, "\n"
	# print " dX = \n", dX, "\n"
	
	D = (c*(1./c).T)/(dX + np.eye(N+1))
	D  = D - np.diag(np.sum(D.T,axis=0))
	x.shape = (N+1,)
	return D, x

