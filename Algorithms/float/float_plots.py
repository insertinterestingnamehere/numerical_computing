import numpy as np
from numpy.random import rand
import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
from matplotlib import pyplot as plt

def sqrt32(A, reps):
    Ac = A.copy()
    I = Ac.view(dtype=np.int32)
    I >>= 1
    I += (1<<29) - (1<<22) - 0x4C000
    for i in xrange(reps):
        Ac = .5 *(Ac + A / Ac)
    return Ac

def sqrt64(A, reps):
    Ac = A.copy()
    I = Ac.view(dtype=np.int64)
    I >>= 1
    I += (1<<61) - (1<<51)
    for i in xrange(reps):
        Ac = .5 *(Ac + A / Ac)
    return Ac

# These do the same thing as the cython functions for the inverse square root.
def invsqrt32(A, reps):
    Ac = A.copy()
    if 0 < reps:
        Ac2 = A.copy()
        Ac2 /= - 2
        Ac3 = np.empty_like(Ac)
    I = Ac.view(dtype=np.int32)
    I >>= 1
    I *= -1
    I += 0x5f3759df #hexadecimal representation of the constant
    for j in xrange(reps):
        Ac3[:] = Ac
        Ac3 *= Ac
        Ac3 *= Ac2
        Ac3 += 1.5
        Ac *= Ac3
    return Ac

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

X = np.linspace(0, 3, 501)
plt.plot(X, sqrt64(X, 0), X, np.sqrt(X))
plt.savefig("sqrt0.pdf")
plt.cla()
plt.plot(X, sqrt64(X, 1), X, np.sqrt(X))
plt.savefig("sqrt1.pdf")
plt.cla()
X = np.linspace(.1, 3, 291)
plt.plot(X, invsqrt64(X, 0), X, 1./np.sqrt(X))
plt.savefig("invsqrt0.pdf")
plt.cla()
plt.plot(X, invsqrt64(X, 1), X, 1./np.sqrt(X))
plt.savefig("invsqrt1.pdf")
plt.cla()
