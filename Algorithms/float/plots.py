import numpy as np
from numpy.random import rand
import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
from matplotlib import pyplot as plt

from solutions import sqrt64


def invsqrt64(A, reps):
    Ac = A.copy()
    if 0 < reps:
        Ac2 = A.copy()
        Ac2 /= - 2
        Ac3 = np.empty_like(Ac)
    I = Ac.view(dtype=np.int64)
    I >>= 1
    I *= -1
    I += 0x5fe6ec85e7de30da #hexadecimal representation of the constant
    for j in xrange(reps):
        Ac3[:] = Ac
        Ac3 *= Ac
        Ac3 *= Ac2
        Ac3 += 1.5
        Ac *= Ac3
    return Ac


def sqrt0():
    X = np.linspace(0, 3, 501)
    plt.plot(X, sqrt64(X, 0), X, np.sqrt(X))
    plt.savefig("sqrt0.pdf")
    plt.clf()


def sqrt1():
    X = np.linspace(0, 3, 501)
    plt.plot(X, sqrt64(X, 1), X, np.sqrt(X))
    plt.savefig("sqrt1.pdf")
    plt.clf()

    
def invsqrt0():
    X = np.linspace(.1, 3, 291)
    plt.plot(X, invsqrt64(X, 0), X, 1./np.sqrt(X))
    plt.savefig("invsqrt0.pdf")
    plt.clf()


def invsqrt1():
    X = np.linspace(.1, 3, 291)
    plt.plot(X, invsqrt64(X, 1), X, 1./np.sqrt(X))
    plt.savefig("invsqrt1.pdf")
    plt.clf()

    
if __name__ == "__main__":
    sqrt0()
    sqrt1()
    invsqrt0()
    invsqrt1()
