import numpy as np
import pyfftw.interfaces.scipy_fftpack as ft
from numpy.polynomial.chebyshev import Chebyshev
from matplotlib import pyplot as plt

def nodes(a, b, n):
    """Get Chebyshev nodes on [a,b]"""
    vals = np.linspace(0, np.pi, n+1)
    np.cos(vals, out=vals)
    vals *= (b - a) / 2.
    vals += (b + a) / 2.
    # reverse order to get them in ascending order.
    return vals[::-1]

# more or less what I'd expect from a student.
def cheb_eval(X, cfs):
    """Evaluate series with coefficients cfs at points X."""
    current = np.empty_like(X)
    prev1 = np.empty_like(X)
    prev2 = np.zeros_like(X)
    size, order = X.size, cfs.size
    prev1[:] = cfs[order-1]
    for k in xrange(order-2, 0, -1):
        current[:] = 2 * X * prev1 - prev2 + cfs[k]
        prev2[:] = prev1
        prev1[:] = current
    # mixing last step with final formula
    current[:] = X * prev1 - prev2 + cfs[0]
    return current

def get_coefs(samples):
    """Compute coefficients of the Chebyshev interpolant
    using the discrete cosine transform."""
    cfs = ft.dct(samples, type=1)
    cfs /= (samples.size - 1)
    cfs[::cfs.size-1] /= 2
    cfs[1::2] *= -1
    return cfs

def cos_interp():
    a = -1
    b = 1
    order = 20
    resolution = 501
    X = nodes(a, b, resolution)
    F = lambda x: np.cos(x)
    A = F(X)
    cfs = get_coefs(A)
    print "number of coefficients for cos that are greater than 1E-14: ", (np.absolute(cfs) > 1E-14).sum()
    f = Chebyshev(cfs)
    X2 = np.linspace(a, b, resolution)
    ax = plt.subplot(1, 1, 1)
    plt.plot(X2, F(X2), label="$\\cos x$")
    plt.plot(X2, f(X2), label="Chebyshev Interpolant")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper right")
    plt.show()

def crazy_interp():
    # This one takes a little time to run
    a = -1
    b = 1
    order = 100000
    resolution = 100001
    # Where to sample.
    X = nodes(a, b, order)
    F = lambda x: np.sin(1./x) * np.sin(1./np.sin(1./x))
    A = F(X)
    cfs = get_coefs(A)
    print "The last 10 coeffients are: ", cfs[-10:]
    f = Chebyshev(cfs)
    # Sample values for plot.
    X2 = np.linspace(a, b, resolution)
    ax = plt.subplot(1, 1, 1)
    plt.plot(X2, f(X2), label="Chebyshev Interpolant")
    plt.plot(X2, F(X2), label="$\\sin\\frac{1}{x} \\sin\\frac{1}{\\sin\\frac{1}{x}}$")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper right")
    plt.show()
