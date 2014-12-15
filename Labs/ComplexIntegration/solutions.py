import numpy as np
from sympy import mpmath as mp
from matplotlib import pyplot as plt

def singular_surface_plot(f, mn=-1., mx=1., res=500, threshold=2., lip=.1):
    x = np.linspace(mn, mx, res)
    X, Y = np.meshgrid(x, x, copy=False)
    Z = f(X + 1.0j * Y)
    
    Z = np.absolute(Z)
    Z[(threshold+lip>Z)&(Z>threshold)] = threshold
    Z[(-threshold-lip<Z)&(Z<-threshold)] = -threshold
    Z[np.absolute(Z) >= threshold + lip] = np.nan
    mlab.mesh(X, Y, Z, color=(1,0,0))
    mlab.show()

def partial_fractions(p, q):
    """ Finds the partial fraction representation of the rational
    function 'p' / 'q' where 'q' is assumed to not have any repeated
    roots. 'p' and 'q' are both assumed to be numpy poly1d objects.
    Returns two arrays. One containing the coefficients for
    each term in the partial fraction expansion, and another containing
    the corresponding roots of the denominators of each term. """
    residues = p(q.roots) / q.deriv()(q.roots)
    return residues, q.roots

def cpv(p, q, tol = 1E-8):
    """ Evaluates the cauchy principal value of the integral over the
    real numbers of 'p' / 'q'. 'p' and 'q' are both assumed to be numpy
    poly1d objects. 'q' is expected to have a degree that is
    at least two higher than the degree of 'p'. Roots of 'q' with
    imaginary part of magnitude less than 'tol' are treated as if they
    had an imaginary part of 0. """
    residues = p(q.roots) / q.deriv()(q.roots)
    return - np.pi * 1.0j * (2 * residues[residues.imag > tol].sum() +
                             residues[np.absolute(residues.imag) <= tol].sum())

def count_roots(p):
    """ Counts the number of roots of the polynomial object 'p' on the
    interior of the unit ball using an integral. """
    return mp.quad(lambda z: mp.exp(1.0j * z) * p(mp.exp(1.0j * z)), [0., 2 * np.pi]) / (2 * np.pi)
