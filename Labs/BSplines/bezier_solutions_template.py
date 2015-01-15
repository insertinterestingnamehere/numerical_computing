import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import comb
from numpy.random import rand

# Decasteljau's algorithm problem
def decasteljau(p,t):
    """ Evaluates a Bezier curve with control points 'p' at time 't'.
    The points in 'p' are assumed to be stored in its rows.
    'p' is assumed to be 2-dimensional."""
    pass

# Bernstein polynomial problem
def bernstein(i, n):
    """ Returns the 'i'th Bernstein polynomial of degree 'n'."""
    pass

# Coordinate function problem.
def bernstein_pt_aprox(X):
    """ Returns the 'x' and 'y' coordinate functions for a 2-dimensional
    Bezier curve with control points 'X'."""
    pass

# plot demonstrating numerical instability in Bernstein polys.
def compare_plot(n, res=501):
    """ Produces a plot showing a Bezier curve evaluated via
    the Decasteljau algorithm and a Bezier curve evaluated using
    Bernstein polynomials. Control points are chosen randomly.
    Instability should be evident for moderately large values of 'n'."""
