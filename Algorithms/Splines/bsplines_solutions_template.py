import numpy as np
from math import isnan, isinf
from scipy.interpolate import splev
from matplotlib import pyplot as plt

# Recursive De Boor algorithm problem.
def N(i, p, t, u, tol=1E-13):
    """ Computes the i'th basis function of order 'p'
    for the spline with knot vector 't' at the parameter value 'u'."""

# Circle approximation problem,
# 'splev' version.
def circle_interp(n, k, res=401):
    """ Plots an interpolating spline of degree 'k'
    with parameters ranging from 1 to 'n'
    that approximates the unit circle.
    Uses scipy.integrate.splev."""

# version using your own code
def my_circle_interp(n, k, res=401):
    """ Plots an interpolating spline of degree 'k'
    with parameters ranging from 1 to 'n'
    that approximates the unit circle.
    Uses the function 'N' defined above."""

# What the script does if it is run from command line.
if __name__=="__main__":
    circle_interp(20, 4)
    my_circle_interp(20, 4)
