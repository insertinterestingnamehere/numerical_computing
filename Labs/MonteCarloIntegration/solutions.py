import numpy as np
from matplotlib import pyplot as plt
 
def mc_int(f, mins, maxs, numPoints=500, numIters=100):
    '''Use Monte-Carlo integration to approximate the integral of f
    on the box defined by mins and maxs.
    
    INPUTS:
    f         - A function handle. Should accept a 1-D NumPy array 
    	 	as input.
    mins      - A 1-D NumPy array of the minimum bounds on integration.
    maxs      - A 1-D NumPy array of the maximum bounds on integration.
    numPoints - An integer specifying the number of points to sample in 
    		the Monte-Carlo method. Defaults to 500.
    numIters - An integer specifying the number of times to run the 
    		Monte Carlo algorithm. Defaults to 100.
		
    ALGORITHM:
    Run the Monte-Carlo algorithm `numIters' times and return the average
    of these runs.
                
    EXAMPLES:
    >>> f = lambda x: np.hypot(x[0], x[1]) <= 1
    >>> # Integral over the square [-1,1] x [-1,1] should be pi
    >>> mc_int(f, np.array([-1,-1]), np.array([1,1]))
    3.1290400000000007
    '''
    side_lengths = maxs-mins
    dims = mins.shape[0]
    
    answers = np.empty(numIters)
    
    for i in xrange(numIters):
        points = np.random.rand( numPoints, dims )
        points = points*side_lengths + mins
        total = np.sum(np.apply_along_axis(f, 1, points))
        answers[i] = np.prod(side_lengths)*float(total)/numPoints
        
    return answers.mean()