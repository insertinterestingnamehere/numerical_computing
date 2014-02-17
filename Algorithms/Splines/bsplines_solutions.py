import numpy as np
from math import isnan, isinf
from scipy.interpolate import splev
from matplotlib import pyplot as plt

# Recursive De Boor algorithm problem.
def N(x, i, k, t, tol=1E-13):
    """ Computes the i'th basis function of order 'k'
    for the spline with knot vector 't' at the parameter value 'x'."""
    # This recursion involves a lot
    # of redundant calculation.
    # This is not the way this algorithm
    # should be implemented in real world
    # applications, but it is instructive.
    # Do k=0 case.
    if k <= 0:
        if t[i] <= x < t[i+1]:
            return 1.
        else:
            return 0.
    # Use recursion for other cases.
    else:
        # Compute left and right hand sides.
        left = (x - t[i]) / (t[i+k] - t[i])
        right = (t[i+k+1] - x) / (t[i+k+1] - t[i+1])
        # Account for nan and inf values.
        if isnan(left) or isinf(left):
            left = 0.
        if isnan(right) or isinf(right):
            right = 0.
        # Perform the recursive call.
        # It could be good to avoid calling
        # recursively for terms we already know
        # will be zero, but this matches
        # more closely with the formula
        # as it is usually written.
        return left * N(x, i, k-1, t) + right * N(x, i+1, k-1, t)

def circle_interp(n, k, res=401):
    """ Plots an interpolating spline of degree 'k'
    with parameters ranging from 1 to 'n'
    that approximates the unit circle.
    Uses scipy.integrate.splev."""
    # Make the knot vector.
    t = np.array([0]*(k) + range(n) + [n]*(k+1))
    # Preallocate the array 'c' of control points.
    c = np.empty((2, n + k + 1))
    c[:,-1] = 0.
    # Construct the circle.
    # Use n + k control points.
    theta = np.linspace(0, 2 * np.pi, n + k)
    np.cos(theta, out=c[0,:-1])
    np.sin(theta, out=c[1,:-1])
    # Generate the sample values to use for plotting.
    X = np.linspace(0, n, res)
    # Evaluate the B-spline at the given points.
    pts = splev(X, (t, c, k))
    # Plot the B-spline and its control points.
    plt.plot(pts[0], pts[1])
    plt.scatter(c[0], c[1])
    plt.show()

def my_circle_interp(n, k, res=401):
    """ Plots an interpolating spline of degree 'k'
    with parameters ranging from 1 to 'n'
    that approximates the unit circle.
    Uses the function 'N' defined above."""
    # Make the knot vector.
    t = np.array([0]*(k) + range(n) + [n]*(k+1))
    # Preallocate the array 'c' of control points.
    c = np.empty((2, n + k))
    # Construct the circle.
    # Use n + k control points.
    theta = np.linspace(0, 2 * np.pi, n + k)
    np.cos(theta, out=c[0])
    np.sin(theta, out=c[1])
    # Generate the sample vaues to use for plotting.
    # Offset just a little to not get to the end of the interval.
    # This makes the plots look identical instead of just similar.
    X = np.linspace(0, n - 1E-10, res)
    # Find the values of each basis function
    Ni = np.array([[N(x, i, k, t) for x in X] for i in xrange(n + k)])
    # Use the points to evaluate the spline, using the basis functions.
    pts = Ni.T.dot(c.T).T
    # Plot the B-spline and its control points.
    plt.plot(pts[0], pts[1])
    plt.scatter(c[0], c[1])
    plt.show()

if __name__=="__main__":
    circle_interp(20, 4)
    my_circle_interp(20, 4)
