import numpy as np
from scipy.misc import factorial

def series_problem_a():
	c = np.arange(70, -1, -1) # original values for n
	c = factorial(2*c) / ((2*c+1) * factorial(c)**2 * 4**c) #series coeff's
	p = np.zeros(2*c.size) # make space for skipped zero-terms
	p[::2] = c # set nonzero polynomial terms to the series coeff's
	P = np.poly1d(p) # make a polynomial out of it
	return 6 * P(.5) #return pi (since pi/6 = arcsin(1/2))

def series_problem_b():
	p = np.arange(20, -1, -1) # original values for n
	p = (-p)**(p-1) / factorial(p) #compute coefficients
	p[-1] = 0. # get rid of NAN in the zero-term
	P = np.poly1d(p) # Make a polynomial
	print P(.25) * np.exp(P(.25)) # test it
	return P(.25) # return the computed value
