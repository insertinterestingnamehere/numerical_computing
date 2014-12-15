import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import norm


import gibbs

class Data(object):
    data = np.genfromtxt('examscores.csv')
    sigma2 = 25.
    mu_0 = 80.
    sigma2_0 = 16.
    alpha = 3.
    beta = 50.
    mu_samples, sigma2_samples = gibbs.gibbs(data, sigma2, mu_0, sigma2_0, alpha, beta)
    
def mu_posterior():
    mukernel = gaussian_kde(Data.mu_samples)

    x_min = min(Data.mu_samples) - 1.
    x_max = max(Data.mu_samples) + 1.
    x = np.arange(x_min, x_max, step=.1)
    plt.plot(x, mukernel(x))
    plt.savefig("mu_posterior.pdf")
    plt.clf()

def sigma2_posterior():
    sigma2kernel = gaussian_kde(Data.sigma2_samples)

    x_min = min(Data.sigma2_samples) - 1.
    x_max = max(Data.sigma2_samples) + 1.
    x = np.arange(x_min, x_max, step=1.)
    plt.plot(x, sigma2kernel(x))
    plt.savefig("sigma2_posterior.pdf")
    plt.clf()
   
   
def predictiveposterior():
    score_samples = np.array([norm.rvs(mu_sample, np.math.sqrt(sigma2_sample))
                            for mu_sample, sigma2_sample in zip(Data.mu_samples, Data.sigma2_samples)])
    score_kernel = gaussian_kde(score_samples)
    x_min = min(score_samples) - 1.
    x_max = max(score_samples) + 1.
    x = np.arange(x_min, x_max, step=1.)
    plt.plot(x, score_kernel(x))
    plt.savefig("predictiveposterior.pdf")
    

if __name__ == "__main__":
    mu_posterior()
    sigma2_posterior()
    predictiveposterior()