import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
from matplotlib import pyplot as plt

import numpy as np
from cvxopt import matrix, solvers
from scipy import linalg as la
import math
from scipy import sparse as spar
from mpl_toolkits.mplot3d import axes3d
import IntPointIISolutions as sol


def tent():
    """
    Find the tent shape of an 20 x 20 grid using the tent pole configuration given in the lab.
    Plot the tent.
    """
    n=20
    
    #create lower bound for the tent surface
    L = np.zeros((n,n))
    L[n/2-1:n/2+1,n/2-1:n/2+1] = .5
    m = [n/6-1, n/6, int(5*(n/6.))-1, int(5*(n/6.))]
    mask1, mask2 = np.meshgrid(m, m)
    L[mask1, mask2] = .3
    
    #solve the quadratic program:
    # min c^T x + .5 x^T Hx
    # st x >= L
    c = -1.*np.ones(n**2)/((n-1)**2)
    H = sol.laplacian(n)
    A = np.eye(n**2)
    
    #initial guess
    x = np.ones((n,n))
    x = x.ravel()
    y = np.ones(n**2)
    l = np.ones(n**2)
    z = sol.qInteriorPoint(H, c, A, L.ravel(), (x,y,l), niter=10, verbose=False).reshape((n,n))
    
    #plot solution surface
    dom = np.arange(n)
    X, Y = np.meshgrid(dom, dom)
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, L,  rstride=1, cstride=1, color='r')
    ax1.set_aspect('equal')
    plt.axis('off')
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, z,  rstride=1, cstride=1, color='b')
    ax2.set_aspect('equal')
    plt.axis('off')
    plt.savefig('tent.pdf')
    plt.clf()


def frontier():
    # Markowitz portfolio optimization
    data = np.loadtxt('portfolio.txt')
    data = data[:,1:]
    mu = 1.

    # calculate covariance matrix
    Q = np.cov(data.T)

    # calculate returns
    R = data.mean(axis=0)

    P = matrix(Q)
    c = matrix(np.zeros(Q.shape[0]))
    A = np.ones((2, Q.shape[0]))
    A[1,:] = R
    A = matrix(A)

    n = 30
    risks = np.empty(n)
    rets = np.empty(n)
    i=0
    for mu in np.linspace(1.05,1.13,n):
        b = np.array([1., mu])
        b = matrix(b)
        sol = solvers.qp(P, c, A = A, b = b)
        risks[i] = math.sqrt(2*sol['primal objective'])
        rets[i] = mu
        i += 1
    plt.plot(risks, rets)
    minrisk = risks.argmin()
    plt.plot(risks[minrisk:], rets[minrisk:], 'r', linewidth=3.0)
    plt.yticks([])
    plt.xticks([])
    plt.xlabel('Risk')
    plt.ylabel('Return')
    plt.annotate('Efficient Frontier', xy=(.04, 1.103), xytext=(.03, 1.12),
                 arrowprops=dict(facecolor='black', shrink=0.02),)
    plt.text(.06, 1.09,'Inefficient Portfolios')
    plt.savefig('frontier.pdf')
    plt.clf()
    
    
if __name__ == "__main__":
    tent()
    frontier()