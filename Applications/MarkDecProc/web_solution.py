# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:05:10 2013

@author: Jeff Hendricks
Web Page Experiments Solutions
"""

import scipy as sp
import bandit_solution as bs
from matplotlib import pyplot as plt


#Problem 1
#simulates one day of web page experiments.  Returns the weights used that day
#and also returns priors (or posteriors really), being the state each each arm
#i.e. the number of successes and failures for each arm (+1)
def simulate_day(pvec, priors):
    
    priors = priors.copy()
    n = pvec.size
    datasize = 100
    
    for k in xrange(0, 2):
        data = bs.sim_data(priors, datasize)
        prob = bs.prob_opt(data)
<<<<<<< HEAD
        weights = bs.get_weights(prob,50)
        for i in xrange(0,n):
            for j in xrange(0,int(weights[i])):
=======
        weights = bs.get_weights(prob, 50)
        for i in xrange(0, n):
            for j in xrange(0, int(weights[i])):
>>>>>>> d96a9c262a9c5e7a28574e4fbcf025329111a043
                result = pull(pvec[i])
                if result == 0:
                    priors[i,1] += 1
                else:
                    priors[i,0] += 1
                    
    return priors,weights

#Simulates a webpage visit resulting in a conversion or not        
def pull(p):
    return sp.random.binomial(1, p, size=None)    
    
    
#Problem 2
#this function is given in the lab.  It computes a measure of "value remaining"
# in the experiment as described in the Google Analytics page
def val_remaining(data,prob):
    champ_ind = sp.argmax(prob)
    thetaM = sp.amax(data,1)
    valrem = (thetaM - data[:,champ_ind])/data[:,champ_ind]
    pvr = sp.stats.mstats.mquantiles(valrem, .95)
    
    return valrem, pvr    
    
#Simulates web page testing until the winning variation is found    
def simulate_convergence(pvec):
    n = pvec.size
    priors = sp.ones((n,2))
    datasize = 100
    
    delta = 0
    p_tol = .95
    champ_cvr = 0
    pvr = 1
    days = 0
    
    weights = []
    while ((delta < p_tol) and (champ_cvr/100. < pvr)) or days < 14:
        days += 1
        priors, weights1 = simulate_day(pvec, priors)
        weights.append(weights1)
        data = bs.sim_data(priors, datasize)
        prob = bs.prob_opt(data)
        delta = prob.max()
        valrem_dist, pvr = val_remaining(data, prob)
        champ_ind = sp.argmax(prob)
        champ_cvr = priors[champ_ind,0]/float(priors[champ_ind,0] + priors[champ_ind,1])            

    return priors,weights,champ_ind,days

#Problem 3
#Create plots for 2-arm convergence
if __name__ == "__main__":
    pvec = sp.array([.04,.05])
    priors, weights, champ_ind, days = simulate_convergence(pvec)
    weights = sp.array(weights)/float(50)
    plt.figure()
    plt.plot(weights)
    plt.plot([0,days],[.95,.95],'--k')
    plt.plot([0,days],[.05,.05],'--k')


    dayvec1 = sp.zeros(200)
    champ1 = sp.zeros(200)
    #run 200 simulations of 2-arm case
    for i in xrange(0,200):
        print(i)
        priors, weights, champ_ind, days = simulate_convergence(pvec)
        dayvec1[i] = days
        champ1[i] = champ_ind
        
    plt.figure()
    hist, bins = sp.histogram(dayvec1,bins = 12)
    width = (bins[1]-bins[0])
    center = (bins[:-1]+bins[1:])/2
    plt.bar(center, hist, align = 'center', width = width,color = 'g')


    #Create plots for 6-arm convergence
    pvec = sp.array([.04,.02,.03,.035,.045,.05])
    priors, weights, champ_ind, days = simulate_convergence(pvec)
    weights = sp.array(weights)/float(50)
    plt.figure()
    plt.plot(weights)
    plt.plot([0,days],[.95,.95],'--k')
    plt.plot([0,days],[.05,.05],'--k')

    dayvec2 = sp.zeros(100)
    champ2 = sp.zeros(100)
    #run 100 simulations of 6-arm case
    for i in xrange(0,100):
        print(i)
        priors, weights, champ_ind, days = simulate_convergence(pvec)
        dayvec2[i] = days
        champ2[i] = champ_ind
        
    plt.figure()
    hist, bins = sp.histogram(dayvec2,bins = 12)
    width = (bins[1]-bins[0])
    center = (bins[:-1]+bins[1:])/2
    plt.bar(center, hist, align = 'center', width = width,color = 'g')

    plt.show()
