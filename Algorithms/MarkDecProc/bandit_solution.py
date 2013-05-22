# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:28:20 2013

@author: Jeff Hendricks
Bandit Solutions
"""
#Problem 1
#simulates data from beta distributions with parameters contained in the 
#matrix priors.  returns k draws from distribution
def sim_data(priors,k):
    import scipy as sp
    n = sp.shape(priors)[0]
    data = sp.zeros((k,n))
    for i in xrange(0,n):
        data[:,i] = sp.random.beta(priors[i,0],priors[i,1],k)
        
    return data

#Problem 2
import scipy as sp    
import scipy.stats
priors = sp.array([[100,200]])
data = sim_data(priors,100)
pbar = data.mean()
q = sp.stats.mstats.mquantiles(data ,.95)

#Problem 3
def prob_opt(data):
    import scipy as sp
    k = sp.shape(data)[0]
    n = sp.shape(data)[1]
    max_arm = sp.argmax(data,axis = 1)
    prob = sp.zeros(n)
    for i in xrange(0,n):
        this_max = max_arm == i
        prob[i] = this_max.sum()/float(k)
        
    return prob
        
#Problem 4
def get_weights(prob,M):
    import scipy as sp
    n = prob.shape[0]
    weights = sp.floor(prob*M)
    missing = sp.int32(M - weights.sum())
    for i in xrange(0,missing):
        assign_index = sp.random.random_integers(0,n-1)
        weights[assign_index] = weights[assign_index]+1
        
    return weights