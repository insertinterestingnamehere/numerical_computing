import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import isingmodel
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
    
    
if __name__ == "__main__":    
    initialize()
    beta(100, beta=1)
    beta(100, beta=.2)
