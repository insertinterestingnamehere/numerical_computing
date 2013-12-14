import numpy as np
from numpy.random import rand
from cmath import sqrt
from scipy.linalg import eig, inv
import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
from matplotlib import pyplot as plt
from solutions import arnoldi

def arnoldi_convergence_plot(A, b, k, view_vals, filename):
    difs = np.empty((view_vals, k))
    A_eigs = eig(A, right=False)
    A_eigs = A_eigs[np.absolute(A_eigs).argsort()[::-1]]
    for i in xrange(1, k+1):
        H = arnoldi(b, A.dot, i)[0]
        H_eigs = eig(H[:-1], right=False)
        H_eigs = H_eigs[np.absolute(H_eigs).argsort()[::-1]]
        difs[:min(view_vals, H_eigs.size),i-1] = np.absolute(H_eigs[:view_vals].real - A_eigs[:min(view_vals,H_eigs.size)].real)
    X = np.arange(2, k+2)
    difs[difs<1E-16] = 1E-16
    for i in xrange(view_vals):
        plt.semilogy(X[i:], difs[i,i:] / np.absolute(A_eigs[i].real))
    plt.xlim((0, k))
    plt.savefig(filename)
    plt.clf()

if __name__=='__main__':
    m = 500
    X = rand(m, m)
    A = np.zeros((m, m))
    np.fill_diagonal(A, rand(m))
    A[:] = X.dot(A).dot(inv(X))
    b = rand(m)
    arnoldi_convergence_plot(A, b, 300, 15, 'rand_eigs_conv.pdf')
    arnoldi_convergence_plot(X, b, 200, 15, 'rand_vals_conv.pdf')
