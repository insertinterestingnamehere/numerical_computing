import numpy as np

def plot_real(f, xbounds=(-1, 1), ybounds=(-1, 1), res=401):
    """ Make a surface plot of the real part
    of the function 'f' given the bounds and resolution. """
    pass

def plot_poly_imag_mayavi(p, xbounds=(-1, 1), ybounds=(-1, 1), res=401):
    """ Plot the imaginary part of the function 'f'
    given the bounds and resolution."""
    pass

def plot_poly_both_mayavi(p, xbounds=(-1, 1), ybounds=(-1, 1), res=401):
    """ Plot the real and imaginary parts of
    the function 'f', given the bounds and resolution."""

def nroot_real_mayavi(n, res=401):
    """ Plot the Riemann surface for the real part
    of the n'th root function."""
    pass

def nroot_imag_mayavi(n, res=401):
    """ Plot the Riemann surface for the imaginary part
    of the n'th root function."""
    pass

def contour_int(f, c, t0, t1):
    """ Evaluate the integral of the function 'f'
    parameterized by the function 'c' with initial
    and final parameter values 't0' and 't1'."""

def cauchy_formula(f, c, z0, t0, t1):
    """ Compute the integral in Cauchy's Integral formula.
    'f' is a callable function parameterized by the contour 'c'.
    'z0' is a point on the interior of 'c'.
    't0' and 't1' are the initial and final parameter values."""
