import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import beta, norm


def bernoulli_posterior(data, prior):
    n_1 = sum(data)
    n_2 = len(data) - n_1
    x = np.arange(0, 1.01, step=.01)
    y = beta.pdf(x, prior[0] + n_1 - 1, prior[1] + n_2 - 1)
    plt.plot(x, y)
    plt.show()


def problem1():
    data = np.fromfile('trial.csv', dtype=int, sep='\n')
    bernoulli_posterior(data, np.array([8., 2.]))


def multinomial_beta(alpha):
    return np.prod(np.array([math.gamma(alpha[i]) for i in xrange(len(alpha))])) / math.gamma(np.sum(alpha))


def multinomial_posterior(data, prior):
    categories = np.sort(list(set(data)))
    n = len(categories)
    sums = np.zeros(n)
    for i in xrange(n):
        sums[i] = sum(data == categories[i])
    return prior + sums


def problem2():
    with open('fruit.csv', 'r') as fruit:
        data = np.array([line.strip() for line in fruit])
    multinomial_posterior(data, np.array([5., 3., 4., 2., 1.]))


def invgammapdf(x, alpha, beta):
    alpha = float(alpha)
    beta = float(beta)
    if not np.isscalar(x):
        return (beta ** alpha / math.gamma(alpha)) * np.array([(xi ** (-alpha - 1)) * math.exp(-beta / xi) for xi in x])
    else:
        return (beta ** alpha / math.gamma(alpha)) * (x ** (-alpha - 1)) * math.exp(-beta / x)


def mean_posterior(data, prior, variance):
    n = float(len(data))
    sample_mean = np.mean(data)
    mu_0 = prior[0]
    sigma2_0 = prior[1]
    posterior_mean = (
        (mu_0 * variance) / n + sample_mean * sigma2_0) / (variance / n + sigma2_0)
    posterior_variance = 1. / (n / variance + 1. / sigma2_0)
    return posterior_mean, posterior_variance


def problem3():
    data = np.fromfile('examscores.csv', dtype=float, sep='\n')

    posterior_mean, posterior_variance = mean_posterior(
        data, np.array([74., 25.]), 36.)
    x = np.arange(0, 100.1, step=0.1)
    y = norm.pdf(x, loc=posterior_mean, scale=np.sqrt(posterior_variance))
    plt.plot(x, y)
    plt.show()


def variance_posterior(data, prior, mean):
    n = float(len(data))
    alpha = prior[0]
    beta = prior[1]
    posterior_alpha = alpha + n / 2.
    posterior_beta = beta + np.dot(data - mean, data - mean) / 2.
    return posterior_alpha, posterior_beta

#posterior_alpha, posterior_beta = variance_posterior(
    #data, np.array([2., 25.]), 62.)
#x = np.arange(0.1, 100.1, step=0.1)
#y = invgammapdf(x, posterior_alpha, posterior_beta)
#plt.plot(x, y)
#plt.show()
