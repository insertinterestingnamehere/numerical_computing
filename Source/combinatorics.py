import scipy as sp

def Combinations(values, k):
    """This function outputs all possible combinations of k elements from the column vector values"""
    n = len(values)
    try:
        values = sp.row_stack(values)
    except:
        raise ValueError, "I need a 2d column array"

    if k > n:
        raise ValueError, "k must be <= %d" % n
    elif k<=0 or k%1 != 0:
        raise ValueError, "k must be > 0"

    #out = sp.array([],ndmin=2)
    if k == 1:
        return values
    else:
        #This loop iterates through all the elements of the values that have at least
        #k elements.  For each element it then calls Combinations(values[i+1:], k-1) which
        #returns combinations of size k-1 for the elements succeeding the current element
        #We do not want to get repeats of combinations
        #print "for i in range(%d)" % (n-(k-1))
        for i in range(n-(k-1)):
            #Calculate the number of possible combinations (to allow proper concatenation
            #in the recursive call
            numCombs = sp.factorial(n-i)/(sp.factorial(k-1)*sp.factorial(n-i-(k-1)))
            combs = Combinations(values[i:], k-1)
            ones = values[i]*sp.ones((numCombs,1))
            #print "ones: %s \t\t combs: %s" % (str(ones.shape), str(combs.shape))
            print combs
            #hstacked = sp.c_[ones, combs]
            #print hstacked
            #return hstacked
            #print "Shapes: %s \t\t %s" % (str(combs.shape), str(ones.shape))
            #return sp.vstack((hstacked))
            #newrow = sp.concatenate((v[i]*sp.ones((k,1)), Combinations(v[i+1:], r-1)))
            #return sp.concatenate((out,newrow))
