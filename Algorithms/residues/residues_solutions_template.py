import numpy as np
from sympy import mpmath as mp
from matplotlib import pyplot as plt

def singplot(f, mn=-.5, mx=.5, res=401, kind='real'):
    """ Plots a function around a singular point.
    'f' is assumed to be a callable function that can be used on an
    array of complex numbers. 'mn' and 'mx' are the maximum and minimum
    values for both the real and imaginary part of the window to be used
    for plotting. 'res' is the resolution to use along each axis.
    'kind' is a keyworkd argument that can take values of 'real', 'imag',
    or 'abs'. It tells the function to plot the real part, imaginary part,
    or the modulus of the complex function 'f' on the given domain.
    In order to scale the colors properly for plotting, first find the
    values you are to plot (the real part, imaginary part, or modulus),
    then take the sine of the natural logarithm of the absolute value of it.
    This will scale it so that your plot will be able to cover a larger
    range of values. Using the sine function will also make it so that
    the colormap used will be able to cycle through its values repeatedly
    in order to represent the increasingly large values of the 'f'. """
    pass

def partial_fractions(p, q):
    """ Finds the partial fraction representation of the rational
    function 'p' / 'q' where 'q' is assumed to not have any repeated
    roots. 'p' and 'q' are both assumed to be numpy poly1d objects.
    Returns two arrays. One containing the coefficients for
    each term in the partial fraction expansion, and another containing
    the corresponding roots of the denominators of each term. """
    pass

def cpv(p, q, tol = 1E-8):
    """ Evaluates the cauchy principal value of the integral over the
    real numbers of 'p' / 'q'. 'p' and 'q' are both assumed to be numpy
    poly1d objects. 'q' is expected to have a degree that is
    at least two higher than the degree of 'p'. Roots of 'q' with
    imaginary part of magnitude less than 'tol' are treated as if they
    had an imaginary part of 0. """
    pass

def count_roots(p):
    """ Counts the number of roots of the polynomial object 'p' on the
    interior of the unit ball using an integral. """
    pass
