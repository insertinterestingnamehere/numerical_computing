from scipy.misc import factorial
def exp(x, n = 25):
	# np.arange(n, -1, -1) starts at n and constructs an array in 
	# reverse order from n to 0.
	p = 1. / factorial(np.arange(n, -1, -1)) # compute coefficients
	P = np.poly1d(p) # make polynomial object
	return P(x)
