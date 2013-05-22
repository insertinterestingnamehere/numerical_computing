#Creates plot for the bandit lab and saves as priors.pdf
import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import scipy as sp
from scipy.stats import beta
import matplotlib.pyplot as plt

#Creates a plot representing Bayesian prior
def priors_plot():
    x = sp.linspace(0,1,1000)
    y2 = beta.pdf(x,1,2)
    y3 = beta.pdf(x,10,20)
    y4 = beta.pdf(x,100,200)
    
    plt.plot(x,y2, label = r'$\alpha = 1$, $\beta = 2$')
    plt.plot(x,y3, label = r'$\alpha = 10$,$\beta = 20$')
    plt.plot(x,y4, label = r'$\alpha = 100$,$\beta = 200$')
    plt.legend()
    plt.savefig('priors.pdf')
    
priors_plot()
