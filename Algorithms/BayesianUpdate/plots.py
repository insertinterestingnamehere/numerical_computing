import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import numpy as np
import solutions
from matplotlib import pyplot as plt
from scipy.stats import beta, norm

def beta_pdf():
    x = np.arange(0, 1.01, .01)
    y = beta.pdf(x, 8, 2)
    plt.plot(x, y)
    plt.savefig("beta_pdf.pdf")
    plt.clf()

def mean_prior():
    x = np.arange(0, 100.1, .1)
    y = norm.pdf(x, loc=74., scale=5)
    plt.plot(x, y)
    plt.savefig("mean_prior.pdf")
    plt.clf()
    

def variance_prior():
    x = np.arange(.1, 100.1, .1)
    y = solutions.invgammapdf(x, alpha=2., beta=25.)
    plt.plot(x, y)
    plt.savefig("variance_prior.pdf")
    plt.clf()


if __name__ == "__main__":
    beta_pdf()
    mean_prior()
    variance_prior()