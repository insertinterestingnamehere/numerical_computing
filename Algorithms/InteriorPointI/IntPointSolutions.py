'''
This is the solutions file for the InteriorPointI lab in volume 2
'''
import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt

def startingPoint(A, b, c):
    '''
    Calculate an initial guess to the solution of the
    linear program min c^T x, Ax = b, x>=0.
    Inputs:
        A -- array of shape (m,n) with linearly independent rows
        b -- array of length m
        c -- array of length n
    Returns:
        x -- array of length n
        lam -- array of length m
        s -- array of length n
    Ref: Nocedal and Wright, p. 410
    '''
    # first calculate x, lam, s of minimal norm satisfying the primal and dual constraints
    B = la.inv(A.dot(A.T))
    x = A.T.dot(B.dot(b))
    lam = B.dot(A.dot(c))
    s = c - A.T.dot(lam)

    # perturb x and s so they are nonnegative
    dx = max((-3./2)*x.min(), 0)
    ds = max((-3./2)*s.min(), 0)
    x += dx*np.ones(x.shape)
    s += ds*np.ones(s.shape)

    # perturb x and s so they are not too close to zero, not too dissimilar
    dx = .5*(x*s).sum()/s.sum()
    ds = .5*(x*s).sum()/x.sum()
    x += dx*np.ones(x.shape)
    s += ds*np.ones(s.shape)

    return x, lam, s

def interiorPoint(A, b, c, niter=20, verbose=False, starting_point=None, pts=False):
    '''
    Solve the linear programming problem min c^T x, Ax = b, x>=0
    using an Interior Point method. This code is not optimized, but
    forms the basis for a common practical approach known as the
    Predictor-Corrector Algorithm.
    Inputs:
        A -- array of shape (m,n) with linearly independent rows
        b -- array of length m
        c -- array of length n
        niter -- positive integer giving the number of iterations
        starting_point -- tuple of arrays giving the initial values for x, l, and s.
                          if unspecified, the function startingPoint is used.
    Returns:
        x -- the optimal point
        val -- the minimum value of the objective function
        (pts -- list of points traced by the algorithm, returned if pts=True)
    Ref: Nocedal and Wright, p. 411
    '''
    pts = []
    # initialize variables
    m,n = A.shape
    if starting_point:
        x, l, s = starting_point
    else:
        x,l,s = startingPoint(A,b,c)
    pts.append(x)
    N = np.zeros((n+m+n, n+m+n))
    N[:n, n:n+m] = A.T
    N[:n, n+m:] = np.eye(n)
    N[n:n+m, :n] = A
    sol = np.empty(n+m+n)
    for k in xrange(niter):
        # finish initializing parts of the step equation
        N[n+m:, :n] = np.diag(s)
        N[n+m:, n+m:] = np.diag(x)
        r_c = (A.T).dot(l)+s-c
        r_b = A.dot(x)-b
        rhs = np.hstack((-r_c.ravel(), -r_b.ravel(), -x*s))

        # solve dx_aff, dl_aff, ds_aff using LU decomposition
        lu_piv = la.lu_factor(N)
        sol[:] = la.lu_solve(lu_piv, rhs)
        dx_aff = sol[:n]
        dl_aff = sol[n:n+m]
        ds_aff = sol[n+m:]

        # calculate a_p, a_d, mu_aff
        mask1 = dx_aff < 0
        if mask1.sum() > 0:
            a_p = min(1, ((-x/dx_aff)[mask1]).min())
        else:
            a_p = 1
        mask2 = ds_aff < 0
        if mask2.sum() > 0:
            a_d = min(1, (-s/ds_aff)[mask2].min())
        else:
            a_d = 1
        mu_aff = ((x+a_p*dx_aff)*(s+a_d*ds_aff)).sum()/np.float(n)

        # calculate mu times the centering parameter sig
        mu = (x*s).sum()/n
        musig = mu_aff**3/mu**2

        # calculate dx, dl, ds
        rhs[n+m:] += - dx_aff*ds_aff + musig
        sol[:] = la.lu_solve(lu_piv, rhs)
        dx = sol[:n]
        dl = sol[n:n+m]
        ds = sol[n+m:]

        # calculate ap, ad
        nu = 1-.1/(k+1)
        mask3 = dx < 0
        if mask3.sum() > 0:
            ap_max = (-x/dx)[mask3].min()
            ap = min(1, nu*ap_max)
        else:
            ap = 1
        mask4 = ds < 0
        if mask4.sum() > 0:
            ad_max = (-s/ds)[mask4].min()
            ad = min(1, nu*ad_max)
        else:
            ad = 1

        # step to new point
        x = x + ap*dx
        l = l + ad*dl
        s = s + ad*ds
        pts.append(x)

        if verbose:
            print '{0:f} {1:f}'.format((c*x).sum(), mu)

    if pts:
        return pts
    else:
        return x, (c*x).sum()

def randomLP(m,n):
    '''
    Generate a linear program min c^T x s.t. Ax = b, x>=0.
    First generate m feasible constraints, then add
    slack variables to convert it into the above form.
    Inputs:
        m -- positive integer >= n, number of desired constraints
        n -- dimension of space in which to optimize
    Outputs:
        A -- array of shape (m,n+m)
        b -- array of shape (m,)
        c -- array of shape (n+m,), with m trailing 0s
        v -- the solution to the LP
    '''
    # generate random constraints (each row corresponds to the normal vector defining
    # a linear constraint)
    A = np.random.random((m,n))*20 - 10

    # adjust so that the normal vector of each constraint lies in the upper half-space.
    # this ensures that the constraints permit a feasible region
    A[A[:,-1]<0] *= -1

    # adjust so that the solution to the program is a prescribed point v in the first
    quadrant.
    v = np.random.random(n)*10
    #k = np.random.randint(n,m+1)
    k = n
    b = np.zeros(m)
    b[:k] = A[:k,:].dot(v)
    b[k:] = A[k:,:].dot(v) + np.random.random(m-k)*10

    # now create the appropriate c vector, a weighted sum of the first k constraints
    c = np.zeros(n+m)
    c[:n] = A[:k,:].sum(axis=0)/k

    # at this point, we should have a program max c^T x s.t. Ax <= b, x >= 0
    # we need to convert it to standard equality form by adding slack variables
    A = np.hstack((A, np.eye(m)))

    # we now have the program min -c^T x s.t. Ax = b, x>=0.
    # the optimal solution has x[:n] = v

    return A, b, -c, v

def leastAbsoluteDeviations():
    """
    This code should be fairly close to what the students submit for the least absolute deviations
    problem.
    """
    data = np.loadtxt('simdata.txt')
    m = data.shape[0]
    n = data.shape[1] - 1
    c = np.zeros(3*m + 2*(n + 1))
    c[:m] = 1
    y = np.empty(2*m)
    y[::2] = -data[:, 0]
    y[1::2] = data[:, 0]
    x = data[:, 1:]
    A = np.ones((2*m, 3*m + 2*(n + 1)))
    A[::2, :m] = np.eye(m)
    A[1::2, :m] = np.eye(m)
    A[::2, m:m+n] = -x
    A[1::2, m:m+n] = x
    A[::2, m+n:m+2*n] = x
    A[1::2, m+n:m+2*n] = -x
    A[::2, m+2*n] = -1
    A[1::2, m+2*n+1] = -1
    A[:, m+2*n+2:] = -np.eye(2*m, 2*m)
    
    sol = interiorPoint(A, y, c, niter=10, verbose=True)[-1]
    beta = (sol[m:m+n] - sol[m+n:m+2*n])[0]
    b = sol[m+2*n] - sol[m+2*n+1]
    
    dom = np.linspace(0,10,2)
    plt.scatter(data[:,1], data[:,0])
    plt.plot(dom, beta*dom+b)
    plt.show()
    print 'Beta:', beta
    print 'b:', b
