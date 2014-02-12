import numpy as np
from matplotlib import pyplot as plt
from math import sqrt, exp
from scipy.linalg import eig
from scipy.integrate import quad

# function shifting problem
def shift_function(f, a, b):
    """ 'f' is a callable funciton, 'a' and 'b' are
    the limits of the interval you want to consider."""
    return lambda x: f((b - a) * x / 2. + (b + a) / 2)

# plotting part of the shifting problem
# The easy way, not the way in the lab.
def funcplot(f, a, b, n=401):
    """ Constructs and plots the example given in the
    problem on shifting the domain of a function to [-1, 1].
    'n' is the number of points to use to generate the plot."""
    X1 = np.linspace(a, b, n)
    X2 = np.linspace(-1, 1, n)
    Y = f(X1)
    plt.plot(X1, Y)
    plt.show()
    plt.plot(X2, Y)
    plt.show()

# alternate version matching what the lab describes
def funcplot2(f, a, b, n=401):
    """ Constructs and plots the example given in the
    problem on shifting the domain of a function to [-1, 1].
    'n' is the number of points to use to generate the plot."""
    X1 = np.linspace(a, b, n)
    g = shift_function(f, a, b)
    plt.plot(X1, f(X1))
    plt.show()
    X2 = np.linspace(-1, 1, n)
    plt.plot(X2, ((b - a) / 2.) * g(X2))
    plt.show()

# example in the function shifting problem
def shift_example(n=401):
    """ Plot the example given in the function shifting problem."""
    f = np.poly1d([1, 0, 0])
    funcplot(f, 1, 4)

# integral estimation problem
def estimate_integral(f, a, b, points, weights):
    """ Estimate the value of an integral given
    the function 'f', the interval bounds 'a' and 'b',
    the nodes to use for sampling, and their
    corresponding weights."""
    g = lambda x: f((b - a) * x / 2. + (b + a) / 2)
    return ((b - a) / 2.) * np.inner(weights, g(points))

# jacobi construction problem
def construct_jacobi(a, b, c):
    """ Construct the Jacobi matrix given the
    sequences 'a', 'b', and 'c' from the 
    three term recurrence relation."""
    alpha = - b / a
    beta = np.sqrt(c[1:] / (a[:-1] * a[1:]))
    i = np.arange(1, a.size+1, dtype=float)
    j = np.zeros((a.size, a.size))
    np.fill_diagonal(j, alpha)
    np.fill_diagonal(j[1:], beta)
    np.fill_diagonal(j[:,1:], beta)
    return j

# points and weights problem
def points_and_weights(n, length):
    """ Find the set of 'n' nodes and their
    corresponding weights for the interval [-1, 1]."""
    i = np.arange(1, n + 1, dtype=float)
    a = (2 * i - 1) / i
    b = np.zeros_like(i)
    c = (i - 1) / i
    j = construct_jacobi(a, b, c)
    # terribly slow to do it this way...
    evals, evects = eig(-j)
    return evals, evects[0]**2 * length

# normal distribution cdf problem
def normal_cdf(x):
    """Compute the CDF of the standard normal
    distribution at the point 'x'."""
    pdf = lambda x: exp(- x**2 / 2.) / sqrt(2 * np.pi)
    return quad(pdf, -5, x)
