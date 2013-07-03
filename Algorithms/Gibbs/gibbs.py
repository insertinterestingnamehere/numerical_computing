import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import invgamma
from scipy.stats import gaussian_kde


def mean_posterior(data, mu_0, sigma2_0, variance):
    n = float(len(data))
    sample_mean = np.mean(data)
    posterior_mean = ((mu_0*variance)/n + sample_mean*sigma2_0)/(
        variance/n + sigma2_0)
    posterior_variance = 1/(n/variance + 1/sigma2_0)
    return posterior_mean, posterior_variance


def variance_posterior(data, alpha, beta, mean):
    n = float(len(data))
    posterior_alpha = alpha + n/2.
    posterior_beta = beta + np.dot(data-mean, data-mean)/2.
    return posterior_alpha, posterior_beta


def sweep(sigma2, mu_0, sigma2_0, alpha, beta, data):
    posterior_mean, posterior_variance = mean_posterior(
        data, mu_0, sigma2_0, sigma2)
    mu = norm.rvs(posterior_mean, scale=np.sqrt(posterior_variance))
    posterior_alpha, posterior_beta = variance_posterior(data, alpha, beta, mu)
    sigma2 = invgamma.rvs(posterior_alpha, scale=posterior_beta)
    return mu, sigma2


def gibbs(data, sigma2, mu_0, sigma2_0, alpha, beta, n_samples=1000):
    mu_samples = np.zeros(n_samples)
    sigma2_samples = np.zeros(n_samples)
    for i in xrange(n_samples):
        mu, sigma2 = sweep(sigma2, mu_0, sigma2_0, alpha, beta, data)
        mu_samples[i] = mu
        sigma2_samples[i] = sigma2
    return mu_samples, sigma2_samples
    
def figures():
    data = np.genfromtxt('examscores.csv')

    sigma2 = 25.
    mu_0 = 80.
    sigma2_0 = 16.
    alpha = 3.
    beta = 50.
    mu_samples, sigma2_samples = gibbs(data, sigma2, mu_0, sigma2_0, alpha, beta)
    
    mukernel = gaussian_kde(mu_samples)

    x_min = min(mu_samples) - 1.
    x_max = max(mu_samples) + 1.
    x = np.arange(x_min, x_max, step=.1)
    plt.plot(x, mukernel(x))
    plt.savefig("mu_posterior.pdf")
    plt.clf()

    sigma2kernel = gaussian_kde(sigma2_samples)

    x_min = min(sigma2_samples) - 1.
    x_max = max(sigma2_samples) + 1.
    x = np.arange(x_min, x_max, step=1.)
    plt.plot(x, sigma2kernel(x))
    plt.savefig("sigma2_posterior.pdf")
    plt.clf()
   
    score_samples = np.array([norm.rvs(mu_sample, np.math.sqrt(sigma2_sample))
                            for mu_sample, sigma2_sample in zip(mu_samples, sigma2_samples)])
    score_kernel = gaussian_kde(score_samples)
    x_min = min(score_samples) - 1.
    x_max = max(score_samples) + 1.
    plt.plot(x, score_kernel(x))
    plt.savefig("predictiveposterior.pdf")
    
figures()
