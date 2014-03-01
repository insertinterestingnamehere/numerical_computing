import numpy as np
from scipy.interpolate import splev
import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
from matplotlib import pyplot as plt

def basis_plot(n, k, res=401):
    """ Plots some b-spline basis functions.
    Uses same knot vector as the circle interpolation problem."""
    # Make the knot vector.
    t = np.array([0]*(k) + range(n) + [n]*(k+1))
    # Preallocate array to store the control points.
    c = np.zeros(t.size - k)
    # Parameter values to use for plot:
    T = np.linspace(0, n, res)
    # Plot each basis function.
    for i in xrange(t.size - k - 1):
        # Set the corresponding coefficient to 1.
        c[i] = 1
        # plot it.
        plt.plot(T, splev(T, (t, c, k)))
        # Set the corresponding coefficient back to 0.
        c[i] = 0.
    # Save and clear the figure.
    plt.savefig("bspline_basis.pdf")
    plt.clf()

if __name__ == "__main__":
    basis_plot(8, 3)
