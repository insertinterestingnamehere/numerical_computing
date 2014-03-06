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
    x = np.linspace(mn, mx, res)
    X, Y = np.meshgrid(x, x, copy=False)
    Z = f(X + 1.0j * Y)
    if kind == 'real':
        Z = Z.real
    elif kind == 'imag':
        Z = Z.imag
    elif kind != 'abs':
        raise ValueError("Kind must be 'real', 'imag' or 'abs'.")
    Z = np.sin(np.log10(np.absolute(Z)))
    Z[np.isnan(Z) | np.isinf(Z)] = 0
    plt.xlim((mn, mx))
    plt.ylim((mn, mx))
    plt.pcolormesh(X, Y, Z)
    plt.show()

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
