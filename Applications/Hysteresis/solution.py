import numpy as np
from scipy.optimize import newton

def EmbeddingAlg(c_list,Initial_Guess,F):
	X = []
	for c in c_list:
		try:
			g = lambda x,c = c: F(x,c)
			Solution = newton(g, Initial_Guess, fprime=None, args=(), tol=1.0e-06, maxiter=80)
			Initial_Guess = Solution # Intial Guess updated for the next iteration
			X.append(Solution)
		except:
			return c_list[:len(X)], X	# returns the list of c values that it was able to find 
	return c_list[:len(X)], X   		# corresponding values of x for, along with those x values						

