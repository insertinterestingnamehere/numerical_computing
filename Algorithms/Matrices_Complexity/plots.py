import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse

def solution1():
    runtimes = [8.95, 36.7, 144, 557]
    inputs = [1000, 2000, 4000, 8000]
    plt.plot(inputs, runtimes, 'go')
    plt.savefig('prob1.pdf')

def spy_sparse():
    n = 10000
    B = np.random.rand(3, n)
    A = sparse.spdiags(B, range(-1, 2), n, n)
    plt.spy(A)
    plt.savefig('spy.pdf')
    
def complexitycurves():
    plt.clf()
    x = np.linspace(.01, 20, 500)
    plt.plot(x, np.log2(x)*x, label='$n\log n$')
    plt.plot(x, x, label='$n$')
    plt.plot(x, x**2, label='$n^2$')
    plt.plot(x, 2**x, label='$2^n$')
    plt.axis([0., 20., 0., 90.])
    plt.xlabel("Problem size (n)")
    plt.ylabel("Execution time")
    plt.legend(loc=2)
    plt.savefig("complexitycurves.pdf")

if __name__ == "__main__":
    spy_sparse()
    complexitycurves()
    solution1()
