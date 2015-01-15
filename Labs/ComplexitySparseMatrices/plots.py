import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse

def upper_bound():
    x = np.linspace(25, 200)
    y1 = 2*x**3
    y2 = (1.5)*x**3+75*x**2+250*x+30
    plt.plot(x, y2, label="f(n)")
    plt.plot(x, y1, label="2n^3")
    plt.legend(loc='upper left')
    plt.savefig('asymp_upper_bound.pdf')
    
def solution1_new():
    x = np.array([100, 200, 400, 800])
    y1 = np.array([.584, 1.17, 2.34, 4.66])
    y2 = np.array([.648, 2.35, 9.05, 35.7])
    y3 = np.array([.592, 2.59, 10.4, 41.2])
    y4 = np.array([.591, 3.05, 19.1, 135])
    y5 = np.array([.579, 2.59, 15.1, 95.5])
    plt.plot(x, y1, label="Function 1")
    plt.plot(x, y2, label="Function 2")
    plt.plot(x, y3, label="Function 3")
    plt.plot(x, y4, label="Function 4")
    plt.plot(x, y5, label="Function 5")
    plt.legend(loc='upper left')
    plt.savefig('complexity_problem.pdf')

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
    upper_bound()
    solution1_new()
