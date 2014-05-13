import numpy as np
import timeit 
def timefunction(f, *args, **kwargs):#f = function name, *args = tuple of function arguments, **kwargs = dictionary of keyword arguments
	pfunc = lambda: f(*args, **kwargs)
	print min(timeit.repeat(pfunc, number=1, repeat=3)
