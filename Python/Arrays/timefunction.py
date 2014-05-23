import numpy as np
import timeit 
# timefunction accepts three arguments: f, the functioon name; *args, 
# a tuple of the function arguments; and **kwargs, a dictionary of the
# keyword arguments.
def timefunction(f, *args, **kwargs):
	pfunc = lambda: f(*args, **kwargs)
	print min(timeit.repeat(pfunc, number=1, repeat=3)
