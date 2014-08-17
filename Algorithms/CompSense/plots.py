import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
from matplotlib import pyplot as plt

import numpy as np
from cvxopt import matrix, solvers
import pyfftw
import math
import scipy.misc
from scipy import linalg as la
import scipy.io as io

def l2Min(A,b):
    """
    Solve min ||x||_2 s.t. Ax = b using CVXOPT.
    Inputs:
        A -- numpy array of shape m x n
        b -- numpy array of shape m
    Returns:
        x -- numpy array of shape n
    """
    m, n = A.shape
    P = np.eye(n)
    q = np.zeros(n)

    P = matrix(P)
    q = matrix(q)
    A = matrix(A)
    b = matrix(b)
    sol = solvers.qp(P,q, A=A, b=b)
    return np.array(sol['x']).flatten()

def l1Min(A,b):
    """
    Solve min ||x||_1 s.t. Ax = b using CVXOPT.
    Inputs:
        A -- numpy array of shape m x n
        b -- numpy array of shape m
    Returns:
        x -- numpy array of shape n
    """
    m, n = A.shape
    A1 = np.zeros((m,2*n))
    A1[:,n:] = A
    c = np.zeros(2*n)
    c[:n] = 1
    h = np.zeros(2*n)
    G = np.zeros((2*n,2*n))
    G[:n,:n] = -np.eye(n)
    G[:n,n:] = np.eye(n)
    G[n:,:n] = -np.eye(n)
    G[n:,n:] = -np.eye(n)

    c = matrix(c)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A1)
    b = matrix(b)

    sol = solvers.lp(c, G, h, A, b)
    return np.array(sol['x'])[n:].flatten()

def sparse():
    # build sparse and nonsparse arrays, plot
    m = 5
    s = np.zeros(100)
    s[np.random.permutation(100)[:m]] = np.random.random(m)
    z = np.random.random_integers(0,high=1,size=(100,100)).dot(s)
    plt.subplot(211)
    plt.plot(s)
    plt.subplot(212)
    plt.plot(z)
    plt.savefig('sparse.pdf')
    plt.clf()

def incoherent():
    # example of sparse image in time domain, diffuse in Fourier domain
    s = np.random.random((50,50))
    mask = s < .98
    s[mask] = 0
    fs = pyfftw.interfaces.scipy_fftpack.fft2(s)
    fs = np.abs(fs)
    plt.subplot(121)
    plt.imshow(1-s,cmap=plt.cm.Greys_r, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.imshow(fs,cmap=plt.cm.Greys_r, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('incoherent.pdf')
    plt.clf()

def reconstruct():
    # reconstruct a simple image

    R = np.zeros((30,30))
    for i in xrange(13):
        R[27-2*i, 2+i] = 1.
        R[27-2*i, -2-i] = 1.
    R[16,9:22] = 1.
    ncols, nrows = R.shape
    n = ncols*nrows
    m = n / 4
    # generate DCT measurement matrix
    D1 = math.sqrt(1./8)*pyfftw.interfaces.scipy_fftpack.dct(np.eye(n), axis=0)[np.random.permutation(n)[:m]]

    # create measurements
    b = D1.dot(R.flatten())

    rec_sig = l1Min(D1,b).reshape((nrows,ncols))
    rec_sig2 = l2Min(D1,b).reshape((nrows,ncols))
    plt.subplot(1,3,1)
    plt.imshow(R, cmap=plt.cm.Greys_r, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,3,2)
    plt.imshow(rec_sig, cmap=plt.cm.Greys_r, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1,3,3)
    plt.imshow(rec_sig2, cmap=plt.cm.Greys_r, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])

    plt.savefig('reconstruct')
    plt.clf()

sparse()
incoherent()
reconstruct()
