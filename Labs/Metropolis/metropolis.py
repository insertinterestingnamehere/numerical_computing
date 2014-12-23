import numpy as np
import scipy.linalg as la
from scipy.stats import bernoulli


def acceptance(x, y, mu, sigma):
    p = min(1, np.exp(-0.5 * (np.dot(x - mu, la.solve(
        sigma, x - mu)) - np.dot(y - mu, la.solve(sigma, y - mu)))))
    return bernoulli.rvs(p)


def nextState(x, mu, sigma):
    K = len(x)
    y = np.random.multivariate_normal(x, np.eye(K))
    accept = acceptance(y, x, mu, sigma)
    if accept:
        return y
    else:
        return x


def lmvnorm(x, mu, sigma):
    return -0.5 * (np.dot(x - mu, la.solve(sigma, x - mu)) - len(x) * np.log(2 * np.pi) - np.log(la.det(sigma)))


def metropolis(x, mu, sigma, n_samples=1000):
    logprobs = np.zeros(n_samples)
    x_samples = np.zeros((n_samples, len(x)))
    for i in xrange(n_samples):
        logprobs[i] = lmvnorm(x, mu, sigma)
        x = nextState(x, mu, sigma)
        x_samples[i, :] = x.copy()
    return x_samples, logprobs
