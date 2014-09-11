import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta


from web_solution import simulate_convergence

#Create plot of 2-arm convergence
def two_arm_plot():
    pvec = np.array([.04, .05])
    priors, weights, champ_ind, days = simulate_convergence(pvec)
    weights = np.array(weights)/50.
    plt.plot(weights)
    plt.plot([0, days], [.95, .95], '--k')
    plt.plot([0, days], [.05, .05], '--k')
    plt.savefig('weights1.pdf')
    plt.clf()
    

#Create plots for 6-arm convergence
def six_arm_plot():
    pvec = np.array([.04, .02, .03, .035, .045, .05])
    priors, weights, champ_ind, days = simulate_convergence(pvec)
    weights = np.array(weights)/50.
    plt.plot(weights)
    plt.plot([0, days], [.95, .95], '--k')
    plt.plot([0, days], [.05, .05], '--k')
    plt.savefig('weights2.pdf')
    plt.clf()
    
    
def priors_plot():
    x = np.linspace(0,1,1000)
    y2 = beta.pdf(x,1,2)
    y3 = beta.pdf(x,10,20)
    y4 = beta.pdf(x,100,200)
    
    plt.plot(x,y2, label = r'$\alpha = 1, \beta = 2$')
    plt.plot(x,y3, label = r'$\alpha = 10, \beta = 20$')
    plt.plot(x,y4, label = r'$\alpha = 100, \beta = 200$')
    plt.legend()
    plt.savefig('priors.pdf')

if __name__ == "__main__":
    two_arm_plot()
    six_arm_plot()
    priors_plot()




