import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import isingmodel
import metropolis
import numpy as np
import scipy.misc as spmisc
import matplotlib.pyplot as plt

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
