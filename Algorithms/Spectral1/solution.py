from __future__ import division
import numpy as np



def cheb(N):
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
