import scipy as sp
import numpy as np
import scipy.linalg as la
import numpy.linalg as la
import matplotlib.pyplot as plt
'''
A = sp.array([[2,1],[4,2.01]])
la.norm(A)*la.norm(la.pinv(A))
np.linalg.cond(A)
'''
#problem 2

def hilbertCond(n): 
	'''
	I'm just going to construct teh Hilbert Matrx and calculate the 
	condition from there. If there's a better way, I don't know it.
	'''
	def hilbert(n):
		return sp.array([[1./(x+y-1) for x in range(1,n+1)] for y in range(1,n+1)])
	return np.linalg.cond (hilbert(n))
'''	
I don't understand the growth curve. When I plot n in ranges like 1 to 21 and 1 to 100 I get strange peaks, but the condition is still obviously growing rapidly. When I plot from 1 to 10 though, I get no peaks. Is this what I'm supposed to get?
'''
if __name__ == "__main__":
	N = sp.arange(1,21)
	plt.plot(N,sp.vectorize(hilbertCond)(N))
	plt.show()

