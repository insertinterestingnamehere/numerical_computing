import numpy as np
from matplotlib import pyplot as plt
from math import sqrt, exp
from scipy.linalg import eig
from scipy.integrate import quad

# plotting problem
# The easy way, not the way in the lab.
def funcplot(f, a, b, n=401):
    Y = f(np.linspace(a, b, n))
    X = np.linspace(-1, 1, n)
    plt.plot(X, Y)
    plt.show()

# alternate version matching what the lab describes
def funcplot2(f, a, b, n=401):
    X = np.linspace(-1, 1, n)
    plt.plot(X, ((b - a) / 2.) * f((b - a) * X / 2. + (b + a) / 2))
    plt.show()

# integral estimation problem
def estimate_integral(f, a, b, points, weights):
    g = lambda x: f((b - a) * x / 2. + (b + a) / 2)
    return ((b - a) / 2.) * np.inner(weights, g(points))

# jacobi construction problem
def construct_jacobi(a, b, c):
    alpha = - b / a
    beta = np.sqrt(c[1:] / (a[:-1] * a[1:]))
    print beta
    i = np.arange(1, a.size+1, dtype=float)
    print np.sqrt(1 / (4 - 1 / i**2))
    j = np.zeros((a.size, a.size))
    np.fill_diagonal(j, alpha)
    np.fill_diagonal(j[1:], beta)
    np.fill_diagonal(j[:,1:], beta)
    return j

# points and weights problem
def points_and_weights(n):
    i = np.arange(1, n + 1, dtype=float)
    a = (2 * i - 1) / i
    b = np.zeros_like(i)
    c = (i - 1) / i
    j = construct_jacobi(a, b, c)
    # terribly slow to do it this way...
    evals, evects = eig(-j)
    return evals, evects[0]

# normal distribution cdf problem
def normal_cdf(x):
    pdf = lambda x: exp(- x**2 / 2.) / sqrt(2 * np.pi)
    return quad(pdf, -5, x)
