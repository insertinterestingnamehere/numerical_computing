from scipy.misc import factorial
def exp(a, n = 25):
	# Construct an array in reverse order from n to 0.
	A = np.arange(n, -1, -1)
	# Use broadcasting to compute coefficients
	p = 1. / factorial(A) 
	P = np.poly1d(p) # make polynomial object
	return P(a)
