import numpy as np
from math import isnan, isinf
from scipy.interpolate import splev

# Recursive De Boor algorithm problem.
def N(i, p, t, u, tol=1E-13):
    # This recursion involves a lot
    # of redundant calculation.
    # This is not the way this algorithm
    # should be implemented in real world
    # applications, but it is instructive.
    # Do p=0 case.
    if p <= 0:
        if u[i] <= t < u[i+1]:
            return 1.
        # Account for last endpoint.
        # This makes the plots look right.
        # Technically, this disagrees with the formula.
        elif t == u[i+1]:
            return 1.
        else:
            return 0.
    # Use recursion for other cases.
    else:
        # Compute left and right hand sides.
        left = (t - u[i]) / (u[i+p] - u[i])
        right = (u[i+p+1] - t) / (u[i+p+1] - u[i+1])
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
        return left * N(i, p-1, t, u) + right * N(i+1, p-1, t, u)
