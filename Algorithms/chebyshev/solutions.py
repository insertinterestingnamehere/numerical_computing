import numpy as np
import pyfftw.interfaces.scipy_fftpack as ft

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
    return cfs
