import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as st
import solutions as sol
from math import sqrt

scores = np.array([98,92,89,77,87,84,75,73,95,86,67,86,86,100,100,92,100,97,95,77,87,87,95,84,84,74,86,84,94])
samples = sol.gibbs(scores,80.,16.,3.,50.,1000)

def mu_posterior():
    mu_ker = st.gaussian_kde(samples[:,0])
    xmin = samples[:,0].min()-1
    xmax = samples[:,0].max()+1
    dom = np.arange(xmin,xmax,.1)
    plt.plot(dom,mu_ker(dom))
    plt.savefig("mu_posterior.pdf")
    plt.clf()

def sigma2_posterior():
    sig_ker = st.gaussian_kde(samples[:,1])
    xmin = samples[:,1].min()-1
    xmax = samples[:,1].max()+1
    dom = np.arange(xmin,xmax,.1)
    plt.plot(dom,sig_ker(dom))
    plt.savefig("sigma2_posterior.pdf")
    plt.clf()    

def predictiveposterior():
    predictive_samples = [st.norm.rvs(s[0], sqrt(s[1])) for s in samples]
    predictive_samples = np.array(predictive_samples)
    # generate and plot the KDE
    score_ker = st.gaussian_kde(predictive_samples)
    xmin = predictive_samples.min()-1
    xmax = predictive_samples.max()+1
    dom = np.arange(xmin,xmax,.1)
    plt.plot(dom, score_ker(dom))
    plt.savefig("predictiveposterior.pdf")
    plt.clf()

if __name__ == "__main__":
    mu_posterior()
    sigma2_posterior()
    predictiveposterior()
