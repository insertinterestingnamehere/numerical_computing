import scipy as sp
from timer import timer

def entropy(values):
    """A slow way to calculate the entropy of the input values"""
    
    values = sp.asarray(values).flatten()
    #calculate the probablility of a value in a vector
    vUni = sp.unique(values)
    vlen = vUni.size
    lenval = float(values.size)
    
    FreqData = sp.zeros_like(vUni)
    for i in range(vlen):
        FreqData[i] = sum(values==vUni[i])/lenval
    print FreqData
    
    return -sum([FreqData[i]*sp.math.log(FreqData[i],2) for i in FreqData])
    
def entropy2(values):
    """Calculate the entropy of vector values.
    
    values will be flattened to a 1d ndarray."""
    
    values = sp.asarray(values).flatten()
    p = sp.diff(sp.c_[sp.diff(sp.sort(values)).nonzero(), values.size])/float(values.size)
    H = (p*sp.log2(p)).sum()
    return -H

def chebyshev2(values, degree=1):
    """Calculate the Chebyshev Polynomials using previous results"""
    
    values = sp.asarray(values)
    A = sp.zeros((degree, len(values)))
    
    A[0,:]=1
    try:
        A[1,:]=values
    except IndexError:
        return A
        
    for i in range(2,degree):
        for x in range(len(values)):
            A[i,x] = 2*values[x]*A[i-1,x]-A[i-2,x]

    return A
    
def chebyshev2_lc(values, degree=1):
    """Calculate the Chebyshev Polynomials using previous results"""
    
    values = sp.asarray(values)
    A = sp.zeros((degree, len(values)))
    
    A[0,:]=1
    try:
        A[1,:]=values
    except IndexError:
        return A
        
    for i in range(2,degree):
        A[i,:] = [2*x for x in values]*A[i-1,:]-A[i-2,:]

    return A
    
def chebyshev_sp(values, degree=1):
    """Calculate the Chebyshev Polynomials using the scipy functions"""
    
    values = sp.asarray(values)
    A = sp.zeros((degree, len(values)))
    
    A[0,:]=1
    try:
        A[1,:]=values
    except IndexError:
        return A
    
    
    for i in range(2,degree): 
        A[i,:] = sp.cos(i*sp.arccos(values))
        
    return A
    
def chebyshev_vec(values, degree=1):
    """Calculate the Chebyshev Polynobials
    
    This implementation uses sp.vectorize to vectorize python's math functions)"""
    
    values = sp.asarray(values)
    A = sp.zeros((degree, len(values)))
    
    A[0,:]=1
    try:
        A[1,:]=values
    except IndexError:
        return A
    
    cos = sp.vectorize(sp.math.cos)
    acos = sp.vectorize(sp.math.acos)
    
    for i in range(2,degree): 
        A[i,:] = cos(i*acos(values))

    return A

def chebyshev_lc(values, degree=1):
    """Calculate the Chebyshev Polynomials using list comprehensions"""
    
    values = sp.asarray(values)
    A = sp.zeros((degree, len(values)))
    
    A[0,:]=1
    try:
        A[1,:]=values
    except IndexError:
        return A
    
    
    for i in range(2,degree): 
        A[i,:] = [sp.math.cos(y) for y in [i*sp.math.acos(x) for x in values]]

    return A
    
def chebyshev(values, degree=1):
    """Calculate the Chebyshev Polynomial using
    
    Tn(x) = cos(n*cosh(x))"""
    values = sp.asarray(values)
    A = sp.zeros((degree, len(values)))
    
    A[0,:]=1
    try:
        A[1,:]=values
    except IndexError:
        return A
    
    
    for i in range(2,degree): 
        for x in values:
            A[i,:] = sp.math.cos(i*sp.math.acos(x))
    return A
    
    
if __name__ == '__main__':
    from timer import timer
    testvals = sp.linspace(-1,1,500)
    funcs = [chebyshev, chebyshev_lc, chebyshev_vec, chebyshev_sp, chebyshev2, chebyshev2_lc]
    with timer(loops=5) as t:
        for f in funcs:
            t.time(f, testvals, 100)
        t.printTimes()
