from scipy.misc import factorial
define exp(x):
	n = 18 # number of terms
	p = 1. / factorial(np.arange(18, -1, -1)) # compute coefficients
	X = np.random.rand(10000) # where to evaluate the series
	P = np.poly1d(p) #make polynomial object
	return P(X)