import numpy as np
from sympy import mpmath as mp
from matplotlib import pyplot as plt

def singular_surface_plot(f, mn=-1., mx=1., res=500, threshold=2., lip=.1):
    """ Plots the absolute value of a function as a surface plot """
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
