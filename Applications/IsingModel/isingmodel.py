import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math

binomial = np.random.binomial
exp = math.exp
randint = np.random.randint

def initialize(n, p=.5):
    state = stats.bernoulli.rvs(p, size=(n ** 2)).astype(float)
    state = state * 2 - 1
    return state.reshape(n, n)

def computeEnergy(state):
    n = state.shape[0]
    energy = 0
    for index, x in np.ndenumerate(state):
        i, j = index
        energy -= x * (state[i-1, j] + state[i, j-1])
    return energy

def proposedEnergy(state, i, j, energy):
    n = state.shape[0]
    S = state[i, j]
    tmp = state[(i - 1) % n, j] + state[(i + 1) % n, j] + state[i, (j - 1) % n] + state[i, (j + 1) % n]
    return energy + 2 * S * tmp


def accept(energy, newEnergy, beta=10):
    if newEnergy <= energy or binomial(1, exp(-beta * (newEnergy - energy))) == 1:
        accepted = True
    return False


def mcmc(n, beta, burnin=100000, n_samples=5000):
    state = initialize(n)
    energy = computeEnergy(state)
    logprobs = np.zeros(burnin + n_samples)
    randvals = np.random.randint(0, n**2, size=(burnin+n_samples))
    print "Burning in ... ({} samples)".format(burnin)
    for k in xrange(burnin):
        logprobs[k] = -beta * energy
        i = randvals[k]/n
        j = randvals[k]%n
        newEnergy = proposedEnergy(state, i, j, energy)
        accepted = accept(energy, newEnergy, beta)
        if accepted:
            state[i, j] *= -1
            energy = newEnergy
    samples = []
    print "Sampling ... ({} samples)".format(n_samples)
    for k in xrange(n_samples):
        logprobs[k + burnin] = -beta * energy
        i = randvals[burnin+k]/n
        j = randvals[burnin+k]%n
        newEnergy = proposedEnergy(state, i, j, energy)
        accepted = accept(energy, newEnergy, beta)
        if accepted:
            state[i, j] *= -1
            energy = newEnergy
        if not k % 100:
            samples.append(state.copy())
    return samples, logprobs
