""" trapSerial.py
    Example usage:
        $ python trapSerial.py 0.0 1.0 10000
        With 10000 trapezoids, the estimate of the integral of x^2 from 0.0 to 1.0 is:
            0.333333335
"""

from __future__ import division
from sys import argv
import numpy as np


def integrate_range(fxn, a, b, n):
    ''' Numerically integrates the function fxn by the trapezoid rule
        Integrates from a to b with n trapezoids
        '''
    # There are n trapezoids and therefore there are n+1 endpoints
    endpoints = np.linspace(a, b, n+1)

    integral = sum(fxn(x) for x in endpoints)
    integral -= (fxn(a) + fxn(b))/2
    integral *= (b - a)/n

    return integral

# An arbitrary test function to integrate
def function(x):
    return x**2

# Read the command line arguments
a = float(argv[1])
b = float(argv[2])
n = int(argv[3])

result = integrate_range(function, a, b, n)
print "With {n} trapezoids, the estimate of the integral of x^2 from {a} to {b} is: \n\t{result}".format(**locals())
