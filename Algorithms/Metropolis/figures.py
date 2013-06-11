import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import matplotlib.pyplot as plt
import numpy as np
import metropolis

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

samples_logs()
