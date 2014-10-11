import numpy as np
from timer import timer

def entropy(values):
    """A slow way to calculate the entropy of the input values"""
    
    values = np.asarray(values).flatten()
    #calculate the probablility of a value in a vector
    vUni = np.unique(values)
    lenval = float(values.size)

    FreqData = np.zeros(vUni.shape, dtype=float)
    for i in xrange(FreqData.size):
        FreqData[i] = sum(values==vUni[i])/lenval

    return -sum([FreqData[i]*np.math.log(FreqData[i],2) for i in xrange(FreqData.size)])
        
def entropy2(values):
    """Calculate the entropy of vector values.
    
    values will be flattened to a 1d ndarray."""
    
    values = np.asarray(values).flatten()
    p = np.diff(np.c_[0, np.diff(np.sort(values)).nonzero(), values.size])/float(values.size)
    H = (p*np.log2(p)).sum()
    return -H

def chebyshev2(values, degree=1):
    """Calculate the Chebyshev Polynomials using previous results"""
    
    values = np.asarray(values)
    A = np.zeros((degree, len(values)))
    
    A[0,:] = 1
    try:
        A[1,:] = values
    except IndexError:
        return A
        
    for i in xrange(2, degree):
        for x in range(len(values)):
            A[i,x] = 2*values[x] * A[i-1,x] - A[i-2,x]

    return A
    
def chebyshev2_lc(values, degree=1):
    """Calculate the Chebyshev Polynomials using previous results"""
    
    values = np.asarray(values)
    A = np.zeros((degree, len(values)))
    
    A[0,:] = 1
    try:
        A[1,:] = values
    except IndexError:
        return A
        
    for i in xrange(2, degree):
        A[i,:] = [2*x for x in values] * A[i-1,:] - A[i-2,:]

    return A
    
def chebyshev_sp(values, degree=1):
    """Calculate the Chebyshev Polynomials using the scipy functions"""
    
    values = np.asarray(values)
    A = np.zeros((degree, len(values)))
    
    A[0,:] = 1
    try:
        A[1,:] = values
    except IndexError:
        return A
    
    
    for i in xrange(2, degree): 
        A[i,:] = np.cos(i * np.arccos(values))
        
    return A
    
def chebyshev_vec(values, degree=1):
    """Calculate the Chebyshev Polynobials
    
    This implementation uses np.vectorize to vectorize python's math functions)"""
    
    values = np.asarray(values)
    A = np.zeros((degree, len(values)))
    
    A[0,:] = 1
    try:
        A[1,:] = values
    except IndexError:
        return A
    
    cos = np.vectorize(np.math.cos)
    acos = np.vectorize(np.math.acos)
    
    for i in xrange(2, degree): 
        A[i,:] = cos(i * acos(values))

    return A

def chebyshev_lc(values, degree=1):
    """Calculate the Chebyshev Polynomials using list comprehensions"""
    
    values = np.asarray(values)
    A = np.zeros((degree, len(values)))
    
    A[0,:] = 1
    try:
        A[1,:] = values
    except IndexError:
        return A
    
    
    for i in xrange(2, degree): 
        A[i,:] = [np.math.cos(y) for y in [i * np.math.acos(x) for x in values]]

    return A
    
def chebyshev(values, degree=1):
    """Calculate the Chebyshev Polynomial using
    
    Tn(x) = cos(n*cosh(x))"""
    values = np.asarray(values)
    A = np.zeros((degree, len(values)))
    
    A[0,:] = 1
    try:
        A[1,:] = values
    except IndexError:
        return A
    
    
    for i in xrange(2, degree): 
        for x in values:
            A[i,:] = np.math.cos(i * np.math.acos(x))
    return A
    
    
if __name__ == '__main__':
    from timer import timer
    testvals = np.linspace(-1,1,500)
    funcs = [chebyshev, chebyshev_lc, chebyshev_vec, chebyshev_sp, chebyshev2, chebyshev2_lc]
    with timer(loops=5) as t:
        for f in funcs:
            t.time(f, testvals, 100)
        t.printTimes()
