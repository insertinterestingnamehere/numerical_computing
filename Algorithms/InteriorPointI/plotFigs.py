import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
from matplotlib import pyplot as plt

import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
import IntPointI_solutions as sol

def interiorPath():
    '''
    Create a plot of a linear program together with the path followed
    by the interior point algorithm.
    '''
    # create a simple linear program
    A = np.array([[5.,1.,-1,0,0],
              [1.5,1.,0,-1,0],
              [.2,1.,0,0,-1]])
    b = np.array([3.,1.8,.75])
    c = -np.array([3.,1., 0, 0, 0])

    # initialize starting point
    x,l,s = sol.startingPoint(-A,-b,c)
    x += 3.5
    s += 3.5

    # solve the program
    pts = sol.interiorPoint(-A,-b,-c, niter=5, starting_point = (x,l,s), pts=True)

    # plot the constraints together with the interior path
    dom = np.linspace(0,10,2)
    for i in xrange(3):
        c = (-A[i,0]*dom + b[i])/A[i,1]
        plt.plot(dom,c,'b')
    pts = np.array(pts)[:,:2]
    plt.plot(pts[:,0], pts[:,1], 'r*-')
    plt.annotate('starting point', xy=(pts[0,0], pts[0,1]), xytext=(4, 3.5),
            arrowprops=dict(facecolor='black', shrink=0.1),)
    plt.annotate('optimal point', xy=(pts[-1,0], pts[-1,1]), xytext=(1.5, 1.5),
                arrowprops=dict(facecolor='black', shrink=0.1),)
    plt.text(2,3,'Feasible Region')
    plt.ylim([0,6])
    plt.xlim([0,6])
    plt.savefig('interiorPath.pdf')
    plt.clf()

#interiorPath()

def leastAbsDev():
    """
    Plot a LAD and LSTSQ line for a set of data.
    """
    #Generate some perturbed linear data
    m = 10
    n = 1
    slope = 3.5
    x = np.random.random(m).reshape((m,1))*10
    x = np.sort(x, axis=0)
    y = slope*x + np.random.randn(m).reshape((m,1))
    y[-1] -= 20 #insert outlier
    
    #Formulate constraint matrix
    A = np.ones((2*m, m+2+2*n+2*m))
    A[::2, :m] = np.eye(m)
    A[1::2, :m] = np.eye(m)
    A[::2, m:m+n] = x
    A[1::2, m:m+n] = -x
    A[::2, m+n:m+2*n] = -x
    A[1::2, m+n:m+2*n] = x
    A[1::2, m+2*n] = -1
    A[::2, m+2*n+1] = -1
    A[:, m+2+2*n:] = -np.eye(2*m, 2*m)
    
    b = np.empty((2*m,1))
    b[::2] = y
    b[1::2] = -y
    b = b.flatten()
    
    c = np.zeros(A.shape[1])
    c[:m] = 1
    
    #Obtain and interpret solution
    pts = sol.interiorPoint(A,b,c, niter=10, verbose=False)[0]
    coeffs = pts[m:m+2*n+2:2] - pts[m+1:m+2*n+2:2]
    
    #Obtain the least squares solution
    B = np.ones((m, n+1))
    B[:,0] = x.flatten()
    coeffs2 = la.lstsq(B, y)[0]
    
    #Plot the data, fitted lines
    dom = np.linspace(0, 10, 2)
    plt.subplot(211)
    plt.scatter(x,y)
    plt.plot(dom, coeffs[0]*dom+coeffs[1])
    plt.subplot(212)
    plt.scatter(x,y)
    plt.plot(dom, coeffs2[0]*dom+coeffs2[1])
    plt.savefig('leastAbsDev.pdf')
    plt.clf()
#leastAbsDev()
