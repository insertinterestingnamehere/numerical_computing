import scipy as sp
import math

#Problem 1
def entropy(values):
    """A slow way to calculate the entropy of the input values"""
    
    values = values.flatten()
    #calculate the probablility of a value in a vector
    vUni = sp.unique(values)
    vlen = len(vUni)
    lenval = float(len(values))
    
    FreqData = sp.zeros_like(vUni)
    for i in range(len(vUni)):
        FreqData[i] = sum(values==vUni[i])/lenval
    
    return -(sum([FreqData[i]*math.log(FreqData[i],2) for i in FreqData]))
    #d={}
    #for i in values:
        #d[i]=values.tolist().count(i)/lenval
    
    #s = 0
    #for i in d.itervalues():
        #s += i*math.log(i, 2)
    #return -s
    
    #return -sum(FreqData)

#Problem 2
def entropy2(values):
    """Calculate the entropy of vector values.
    
    values will be flattened to a 1d ndarray."""
    
    values = values.flatten()
    M = len(sp.unique(values))
    p = sp.diff(sp.c_[sp.diff(sp.sort(values)).nonzero(), len(values)])/float(len(values))
    H = -((p*sp.log2(p)).sum())
    return H
