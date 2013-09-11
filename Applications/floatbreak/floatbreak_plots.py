import numpy as np
from numpy.random import rand
import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
from matplotlib import pyplot as plt

ax = plt.subplot(1,1,1)
X = np.linspace(-5, 30, 17501)
Y = np.log(10**(-X) + 1) / 10**(-X)
plt.plot(X, Y, label="$10^x \\log(10^{-x}+1)$")
X = np.linspace(1, 30, 291)
poly = np.poly1d([(-1.)**(i+1)/i for i in xrange(1,30)][::-1])
Y = poly(10**(-X))
plt.plot(X, Y, label="Series Approximation")
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, labels, loc="upper left")
plt.savefig("lnseries.pdf")
plt.cla()

def pnorm(x, p):
    P = p.reshape((1, -1))
    X = x.reshape((-1, 1))
    return ((X**P).sum(axis=0)**(1./P)).ravel()

ax = plt.subplot(1,1,1)
powers = np.linspace(1, 7, 701)
X = rand(100)+1
N = pnorm(X, 10**powers)
plt.plot([1,7], [X.max()]*2, label="$\\|x\\|_{\\infty}$")
plt.plot(powers, N, label="p-norms of $x$")
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, labels, loc="upper right")
plt.xlabel("$\\log_{10} p$")
plt.ylabel("$\\| x \\|_p$")
plt.savefig("pnorm_convergence.pdf")
plt.cla()
