import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import numpy as np
import matplotlib.pyplot as plt
import gmm
from scipy import stats
from scipy import optimize as opt


def makegrid(data):
    mins = np.min(data, 0)
    maxes = np.max(data, 0)
    x = np.arange(mins[0] - .01, maxes[0] + .01, .0005)
    y = np.arange(mins[1] - .01, maxes[1] + .01, .0005)
    X, Y = np.meshgrid(x, y)
    return X, Y


def baltimore_gmm(data):
    def fgmm(x):
        return abs(np.sum(gmmimmat[gmmimmat > x]) * .0005 ** 2 - 0.95)

    model = gmm.GMM(3)
    model.train(data, random=False)

    X, Y = makegrid(data)
    gmmimmat = np.zeros(X.shape)

    for i in xrange(X.shape[0]):
        for j in xrange(X.shape[1]):
            gmmimmat[i, j] = model.dgmm(np.array([X[i, j], Y[i, j]]))

    plt.jet()
    plt.imshow(gmmimmat, origin='lower')
    plt.ylim([0, X.shape[0]])
    plt.xlim([0, X.shape[1]])
    plt.savefig('baltimore_gmm.pdf')

    thresh = opt.fmin(fgmm, 10)[0]
    bools = gmmimmat > thresh
    mat = np.zeros(X.shape)
    mat += bools
    plt.imshow(mat, origin='lower')
    plt.ylim([0, X.shape[0]])
    plt.xlim([0, X.shape[1]])
    plt.savefig('gmmavoid.pdf')


def baltimore_kde(data):
    def fkde(x):
        return abs(np.sum(kdeimmat[kdeimmat > x]) * .0005 ** 2 - 0.95)

    X, Y = makegrid(data)
    kdeimmat = np.zeros(X.shape)
    kernel = stats.gaussian_kde(data.T)
    for i in xrange(X.shape[0]):
        for j in xrange(X.shape[1]):
            kdeimmat[i, j] = kernel.evaluate(np.array([X[i, j], Y[i, j]]))

    plt.jet()
    plt.imshow(kdeimmat, origin='lower')
    plt.ylim([0, X.shape[0]])
    plt.xlim([0, X.shape[1]])
    plt.savefig('baltimore_kde.pdf')

    thresh = opt.fmin(fkde, 10)[0]
    bools = kdeimmat > thresh
    mat = np.zeros(X.shape)
    mat += bools
    plt.imshow(mat, origin='lower')
    plt.ylim([0, X.shape[0]])
    plt.xlim([0, X.shape[1]])
    plt.savefig('kdeavoid.pdf')

data = np.load('homicides.npy')
baltimore_gmm(data)
baltimore_kde(data)
