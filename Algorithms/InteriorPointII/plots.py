import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
from matplotlib import pyplot as plt

import numpy as np
from cvxopt import matrix, solvers
from scipy import linalg as la
import math
from scipy import sparse as spar
from mpl_toolkits.mplot3d import axes3d

def stepSize(x, y):
    '''
    Return the step size a satisfying max{0 < a <= 1 | x+ay>=0}.
    Inputs:
        x -- numpy array of length n with nonnegative entries
        y -- numpy array of length n
    Returns:
        the appropriate step size.
    '''
    mask = y<0
    if np.any(mask):
        return min(1, (-x[mask]/y[mask]).min())
    else:
        return 1

def startingPoint(G, c, A, b, guess):
    """
    Obtain an appropriate initial point for solving the QP
    .5 x^T Gx + x^T c s.t. Ax >= b.
    Inputs:
        G -- symmetric positive semidefinite matrix shape (n,n)
        c -- array of length n
        A -- constraint matrix shape (m,n)
        b -- array of length m
        guess -- a tuple of arrays (x, y, l) of lengths n, m, and m, resp.
    Returns:
        a tuple of arrays (x0, y0, l0) of lengths n, m, and m, resp.
    """
    m,n = A.shape
    x0, y0, l0 = guess

    # initialize linear system
    N = np.zeros((n+m+m, n+m+m))
    N[:n,:n] = G
    N[:n, n+m:] = -A.T
    N[n:n+m, :n] = A
    N[n:n+m, n:n+m] = -np.eye(m)
    N[n+m:, n:n+m] = np.diag(l0)
    N[n+m:, n+m:] = np.diag(y0)
    rhs = np.empty(n+m+m)
    rhs[:n] = -(G.dot(x0) - A.T.dot(l0)+c)
    rhs[n:n+m] = -(A.dot(x0) - y0 - b)
    rhs[n+m:] = -(y0*l0)

    sol = la.solve(N, rhs)
    dx = sol[:n]
    dy = sol[n:n+m]
    dl = sol[n+m:]

    y0 = np.maximum(1, np.abs(y0 + dy))
    l0 = np.maximum(1, np.abs(l0+dl))

    return x0, y0, l0

def qInteriorPoint(G, c, A, b, guess, niter=20, verbose=False):
    '''
    Solve min .5x^T Gx + x^T c s.t. Ax >= b using a Predictor-Corrector
    Interior Point method.
    Inputs:
        G -- symmetric positive semidefinite matrix shape (n,n)
        A -- constraint matrix size (m,n)
        b -- array of length m
        c -- array of length n
        niter -- integer, giving number of iterations to run
        verbose -- boolean, indicating whether to output print statements
        guess -- tuple of three arrays (x, y, l) of length n, m, and m, an initial estimate
    Returns:
        x -- an array of length n, the minimizer of the quadratic program.
    '''
    # initialize variables
    m,n = A.shape
    x, y, l = startingPoint(G,c,A,b,guess)

    # initialize linear system
    N = np.zeros((n+m+m, n+m+m))
    N[:n,:n] = G
    N[:n, n+m:] = -A.T
    N[n:n+m, :n] = A
    N[n:n+m, n:n+m] = -np.eye(m)
    sol = np.empty(n+m+m)
    rhs = np.empty(n+m+m)

    for i in xrange(niter):
        # finish initializing linear system
        N[n+m:, n:n+m] = np.diag(l)
        N[n+m:, n+m:] = np.diag(y)
        rhs[:n] = -(G.dot(x) - A.T.dot(l)+c)
        rhs[n:n+m] = -(A.dot(x) - y - b)
        rhs[n+m:] = -(y*l)

        # solve dx_aff, dy_aff, dl_aff using LU decomposition
        lu_piv = la.lu_factor(N)
        sol[:] = la.lu_solve(lu_piv, rhs)
        dx_aff = sol[:n]
        dy_aff = sol[n:n+m]
        dl_aff = sol[n+m:]

        # calculate centering parameter
        mu = (y*l).sum()/m
        ahat_aff1 = stepSize(y, dy_aff)
        ahat_aff2 = stepSize(l, dl_aff)
        ahat_aff = min(ahat_aff1,ahat_aff2)
        mu_aff = (y+ahat_aff*dy_aff).dot(l+ahat_aff*dl_aff)/m
        sig = (mu_aff/mu)**3

        # calculate dx, dy, dl
        rhs[n+m:] -= dl_aff*dy_aff - sig*mu
        sol[:] = la.lu_solve(lu_piv, rhs)
        dx = sol[:n]
        dy = sol[n:n+m]
        dl = sol[n+m:]

        # calculate step size
        t = 0.95 # there are other ways to choose this parameter
        ap = stepSize(t*y, dy)
        ad = stepSize(t*l, dl)
        a = min(ap, ad)

        # calculate next point
        x += a*dx
        y += a*dy
        l += a*dl

        if verbose:
            print '{0:f} {1:f}'.format(.5*(x* G.dot(x)).sum() + (x*c).sum(), mu)
    return x

def laplacian(n):
    """
    Construct the discrete Dirichlet energy matrix H for an n x n grid.
    Inputs:
        n -- side length of grid
    Returns:
        dense array of shape n^2 x n^2
    """
    n = n+2
    data = -1*np.ones((5, (n-2)**2))
    data[2,:] = 4
    data[1, n-3::n-2] = 0
    data[3, ::n-2] = 0
    diags = np.array([-n+2, -1, 0, 1, n-2])
    return spar.spdiags(data, diags, (n-2)**2, (n-2)**2).todense()

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
    H = laplacian(n)
    A = np.eye(n**2)
    
    #initial guess
    x = np.ones((n,n))
    x = x.ravel()
    y = np.ones(n**2)
    l = np.ones(n**2)
    z = qInteriorPoint(H, c, A, L.ravel(), (x,y,l), niter=10, verbose=False).reshape((n,n))
    
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
tent()

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
frontier()
    
