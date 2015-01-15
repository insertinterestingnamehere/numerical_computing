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
    
