import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import matplotlib.pyplot as plt
import scipy as sp
import bandit_solution as bs

#===============================================================================
# SOLUTION STUFF USED TO CREATE PLOTS
#===============================================================================
def simulate_day(pvec, priors):
    
    priors = priors.copy()
    n = pvec.size
    datasize = 100
    
    for k in xrange(0,2):
        data = bs.sim_data(priors, datasize)
        prob = bs.prob_opt(data)
        weights = bs.get_weights(prob, 50)
        for i in xrange(0, n):
            for j in xrange(0, sp.int32(weights[i])):
                result = pull(pvec[i])
                if result == 0:
                    priors[i,1] += 1
                else:
                    priors[i,0] += 1
                    
    return priors, weights

#Simulates a webpage visit resulting in a conversion or not        
def pull(p):
    return sp.random.binomial(1, p, size=None)    
    
    
#Problem 2
#this function is given in the lab.  It computes a measure of "value remaining"
# in the experiment as described in the Google Analytics page
def val_remaining(data, prob):
    
    champ_ind = sp.argmax(prob)
    thetaM = sp.amax(data, 1)
    valrem = (thetaM - data[:,champ_ind])/data[:,champ_ind]
    pvr = sp.stats.mstats.mquantiles(valrem, .95)
    
    return valrem, pvr    
    
#Simulates web page testing until the winning variation is found    
def simulate_convergence(pvec):
    n = pvec.size
    priors = sp.ones((n, 2))
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
        valrem_dist,pvr = val_remaining(data, prob)
        champ_ind = sp.argmax(prob)
        champ_cvr = priors[champ_ind,0]/float(priors[champ_ind,0]+priors[champ_ind,1])            

    return priors, weights, champ_ind, days
    
#===============================================================================
#===============================================================================


#Create plot of 2-arm convergence
def two_arm_plot():
    pvec = sp.array([.04, .05])
    priors, weights, champ_ind, days = simulate_convergence(pvec)
    weights = sp.array(weights)/50.
    plt.figure()
    plt.plot(weights)
    plt.plot([0, days], [.95, .95], '--k')
    plt.plot([0, days], [.05, .05], '--k')
    plt.savefig('weights1.pdf')

#Create plots for 6-arm convergence
def six_arm_plot():
    pvec = sp.array([.04, .02, .03, .035, .045, .05])
    priors, weights, champ_ind, days = simulate_convergence(pvec)
    weights = sp.array(weights)/50.
    plt.figure()
    plt.plot(weights)
    plt.plot([0, days], [.95, .95], '--k')
    plt.plot([0, days], [.05, .05], '--k')
    plt.savefig('weights2.pdf')

two_arm_plot()
six_arm_plot()



