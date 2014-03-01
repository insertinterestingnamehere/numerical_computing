import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import matplotlib.pyplot as plt
import scipy as sp

from web_solution import simulate_convergence

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



