import numpy as np
from numpy.random import rand
import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
from matplotlib import pyplot as plt

def pnorm(x, p):
    P = p.reshape((1, -1))
    X = x.reshape((-1, 1))
    return ((X**P).sum(axis=0)**(1./P)).ravel()

powers = np.linspace(1, 7, 701)
X = rand(100)+1
N = pnorm(X, 10**powers)
plt.plot([1,7], [X.max()]*2)
plt.plot(powers, N)
plt.xlabel("$\\log_{10} p $")
plt.ylabel("$\\| x \\|_p $")
plt.savefig("pnorm_convergence.pdf")
plt.cla()
