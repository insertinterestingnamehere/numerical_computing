# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 19:18:03 2013

@author: Jeff Hendricks
"""
#================================================
#Computes discrete approximation to the lognormal distribution
#where xpoints are the points of approximation, m is the mean
#of the lognormal variable (NOT of the normally distributed variable),
#and v is the variance of the lognormal variable
#The function returns discrpdf which gives a vector of
#the probabilities associateed with the xpoints
#================================================
import scipy as sp
from scipy.stats import lognorm
def discretelognorm(xpoints,m,v):
    
    mu = sp.log(m**2/float(sp.sqrt(v+m**2)))
    sigma = sp.sqrt(sp.log((v/float(m**2))+1))    
    
    xmax  = sp.amax(xpoints) 
    xmin  = sp.amin(xpoints) 
    N     = sp.size(xpoints) 
    xincr = (xmax - xmin)/float(N-1)

    binnodes = sp.arange(xmin+.5*xincr,xmax + .5*xincr, xincr)
    lnormcdf  = lognorm.cdf(binnodes,sigma,0,sp.exp(mu)) 
    discrpdf  = sp.zeros((N,1))
    for i in sp.arange(N):
        if i == 0:
            discrpdf[i] = lnormcdf[i] 
        elif (i > 0) and (i < N-1):
            discrpdf[i] = lnormcdf[i] - lnormcdf[i-1] 
        elif (i == N-1):
            discrpdf[i] = discrpdf[i-1]

    return discrpdf
    