import numpy as np
import scipy.linalg as la

def mc_int(f, mins, maxs, numPoints=500):
    '''Use Monte-Carlo integration to approximate the integral of f
    on the box defined by mins and maxs.
    
    INPUTS:
    f         - A function handle. Should accept a 1-D NumPy array as input.
    mins      - A 1-D NumPy array of the minimum bounds on integration.
    maxs      - A 1-D NumPy array of the maximum bounds on integration.
    numPoints - An integer specifying the number of points to sample in 
                the Monte-Carlo method. Defaults to 500.
                
    EXAMPLES:
    >>> f = lambda x: np.hypot(x[0], x[1]) <= 1
    >>> mc_int(f, np.array([-1,-1]), np.array([1,1]))
    3.3199999999999998
    '''
    side_lengths = maxs-mins
    dims = mins.shape[0]
    points = np.random.rand( numPoints, dims )
    points = points*side_lengths + mins
    
    total = np.sum(np.apply_along_axis(f, 1, points))
    return np.prod(side_lengths)*float(total)/numPoints
            

