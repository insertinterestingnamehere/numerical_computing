import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import isingmodel
import metropolis
import numpy as np
import scipy.misc as spmisc
import matplotlib.pyplot as plt
from math import sqrt, exp, log
import scipy.stats as st
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

scores = np.array([98,92,89,77,87,84,75,73,95,86,67,86,86,100,100,92,100,97,95,77,87,87,95,84,84,74,86,84,94])

def initialize():
    spinconfig = isingmodel.initialize(100)
    spmisc.imsave("init.pdf", spinconfig)
    
def beta(n, beta=1):
    samples, logprobs = isingmodel.mcmc(n, beta, n_samples=5000)
    
    stem = str(beta).replace(".", "_")
    plt.plot(logprobs)
    plt.savefig("beta" + stem + "_logprobs.pdf")
    plt.clf()
    spmisc.imsave("beta" + stem + ".pdf", samples[-1])

def samples_logs():
    x = np.array([100., 100.])
    mu = np.zeros(2)
    sigma = np.array([[12., 10.], [10., 16.]])
    samples, logs = metropolis.metropolis(x, mu, sigma, n_samples=2500)
    plt.plot(samples[:,0], samples[:,1], '.')
    plt.savefig('samples.pdf')
    plt.clf()
    
    plt.plot(logs)
    plt.savefig('logprobs.pdf')
    plt.clf()

# let's do a metropolis sampler for the exam scores data
alpha=3
beta=50
mu0 = 80
sig20 = 16
muprior=st.norm(loc=mu0, scale=sqrt(sig20))
sig2prior = st.invgamma(alpha,scale=beta)
def proposal(y, std):
    return st.multivariate_normal.rvs(mean=y, cov=std*np.eye(len(y)))
def propLogDensity(x):
    return muprior.logpdf(x[0])+sig2prior.logpdf(x[1])+st.norm.logpdf(scores,loc=x[0],scale=sqrt(x[1])).sum()
def metropolis(x0, s, n_samples):
    """
    Use the Metropolis algorithm to sample from posterior.
    
    Parameters
    ----------
    x0 : ndarray of shape (2,)
        The first entry is mu, the second entry is sigma2
    s : float > 0
        The standard deviation parameter for the proposal function
    n_samples : int
        The number of samples to generate
        
    Returns
    -------
    draws : ndarray of shape (n_samples, 2)
        The MCMC samples
    logprobs : ndarray of shape (n_samples)
        The log density of the samples
    accept_rate : float
        The proportion of proposed samples that were accepted
    """
    accept_counter = 0
    draws = np.empty((n_samples,2))
    logprob = np.empty(n_samples)
    x = x0.copy()
    for i in xrange(n_samples):
        xprime = proposal(x,s)
        u = np.random.rand(1)[0]
        if log(u) <= propLogDensity(xprime) - propLogDensity(x):
            accept_counter += 1
            x = xprime
        draws[i] = x
        logprob[i] = propLogDensity(x)
    return draws, logprob, accept_counter/float(n_samples)

def traces():
    plt.plot(draws[:,0])
    plt.savefig("mu_traces.pdf")
    plt.clf()
    plt.plot(draws[:,1])
    plt.savefig("sig_traces.pdf")
    plt.clf()
    
def logprobs():
    plt.plot(lprobs[:500])
    plt.savefig("logprobs.pdf")
    plt.clf()
    
def kdes():
    mu_kernel = gaussian_kde(draws[50:,0])
    x_min = min(draws[50:,0]) - 1
    x_max = max(draws[50:,0]) + 1
    x = np.arange(x_min, x_max, step=0.1)
    plt.plot(x,mu_kernel(x))
    plt.savefig("mu_kernel.pdf")
    plt.clf()
    
    sig_kernel = gaussian_kde(draws[50:,1])
    x_min = 20
    x_max = 200
    x = np.arange(x_min, x_max, step=0.1)
    plt.plot(x,sig_kernel(x))
    plt.savefig("sig_kernel.pdf")
    plt.clf()

if __name__ == "__main__":
    draws, lprobs, r = metropolis(np.array([40, 10], dtype=float), 20., 10000)
    traces()
    logprobs()
    kdes()
