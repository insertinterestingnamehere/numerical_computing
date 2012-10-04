import scipy as sp

def Combinations(values, k):
    """This function outputs all the possible combinations of k elements from the vector values"""
    
    if int(k) < 0:
        raise ValueError("k must a positive integer")
    
    #Make input vectors column vectors
    if values.shape == (1,values.size):
        values = sp.atleast2d(values).T.copy()
    
    out = sp.array([]).reshape(0,1)
    n = max(values.shape)
    if k == 1:
        out = values
    else:
        #the following loop interates through all the elements of the vector values that have at least k elements after them.  For each element it then calls Combinations(values[i+1:], k-1) which returns combinations of size k-1 for the elements succeeding the current element.  This is so that we do not get repeats of combinations
        #nck = sp.misc.comb(n,k, exact=True)
        #out = sp.zeros((nck, k))
        for i in range(n-(k-1)):
            #Calculate the number of possible combinations (to allow proper concatenation in the recursive call
            nCombs = int(sp.misc.factorial(n-i)/(sp.misc.factorial(k-1)*sp.misc.factorial(n-i-(k-1))))
            #This is the recursive call
            
            print Combinations(values[i+1:], k-1).reshape((-1,1))
            out = sp.r_[out, sp.c_[values[i]*sp.ones((nCombs,1)), Combinations(values[i+1:], k-1).reshape(-1,1)]]
            
    return out