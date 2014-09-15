from scipy.misc import factorial
def exp(a, n = 25):
	# Construct an array in reverse order from n to 0.
	integers = np.arange(n, -1, -1)
	# Use broadcasting to compute coefficients
	coefficients = 1. / factorial(integers) 
	poly = np.poly1d(coefficients) # make polynomial object
	return poly(a)
