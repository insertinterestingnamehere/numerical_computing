import scipy as sp
import math
from timer import timer

@profile
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
    
@profile
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
        #tmp = [2*x for x in values]
        A[i,:] = [2*x for x in values]*A[i-1,:]-A[i-2,:]

    return A
    
@profile
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
    
@profile
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
    
    cos = sp.vectorize(math.cos)
    acos = sp.vectorize(math.acos)
    
    for i in range(2,degree): 
        A[i,:] = cos(i*acos(values))

    return A
    

@profile
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
        A[i,:] = [math.cos(y) for y in [i*math.acos(x) for x in values]]

    #[A[i,:] = [math.cos(y) for y in [i*math.acos(x) for x in values]] for i in range(3,degree)]
    return A
    
@profile
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
            A[i,:] = math.cos(i*math.acos(x))
              
        #A[i,:] = [math.cos(y) for y in [i*math.acos(x) for x in values]]

    #[A[i,:] = [math.cos(y) for y in [i*math.acos(x) for x in values]] for i in range(3,degree)]
    return A
    
    
if __name__ == '__main__':
    from timer import timer
    testvals = sp.linspace(-1,1,100)
    funcs = [chebyshev, chebyshev_lc, chebyshev_vec, chebyshev_sp, chebyshev2, chebyshev2_lc]
    with timer(loops=5, gc=False) as t:
        for f in funcs:
            t.time(f, testvals, 100)
        t.printTimes()
        #print t.results
