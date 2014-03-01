import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
from solutions import nodes, get_coefs
from matplotlib import pyplot as plt

def node_project():
    resolution = 513
    num_of_nodes = 12
    X = nodes(-1, 1, resolution-1)
    Y = lambda x: np.sqrt(1 - x**2)
    Xpts = nodes(-1, 1, num_of_nodes-1)
    Ypts = Y(Xpts)
    plt.plot(X, Y(X), zorder=-1)
    plt.plot(X, np.zeros_like(X), zorder=-1)
    for x, y in zip(Xpts, Ypts):
        plt.plot([x, x], [y, 0], 'r', zorder=-1)
    plt.scatter(Xpts, Ypts, color='b')
    plt.scatter(Xpts, np.zeros_like(Xpts), color='b')
    plt.xlim((-1.05, 1.05))
    plt.ylim((-.25, 1.25))
    plt.savefig("node_project.pdf")
    plt.clf()

def runge_plot():
    a, b = -1, 1
    resolution = 513
    runge = lambda x: 1 / (1 + 25 * x**2)
    X2 = np.linspace(a, b, resolution)
    ax = plt.subplot(1, 1, 1)
    plt.plot(X2, runge(X2), label="$\\frac{1}{1+25x^2}$")
    for order in [5, 10, 15, 25]:
        X = nodes(a, b, order)
        Y = runge(X)
        C = get_coefs(Y)
        f = Chebyshev(C)
        plt.plot(X2, f(X2), label="Order " + str(order))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper right")
    plt.savefig("runge_chebyshev.pdf")
    plt.clf()

def cheb_polys():
    resolution = 513
    X = np.linspace(-1, 1, resolution+1)
    n = 6
    coeffs = np.zeros(n)
    ax = plt.subplot(1, 1, 1)
    for i in xrange(n):
        coeffs[i] = 1
        f = Chebyshev(coeffs)
        plt.plot(X, f(X), label="$T_" + str(i) + "$")
        coeffs[i] = 0
    plt.ylim((-1.2,1.2))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper right")
    plt.savefig("cheb_polys.pdf")
    plt.clf()

if __name__ == "__main__":
    node_project()
    runge_plot()
    cheb_polys()
