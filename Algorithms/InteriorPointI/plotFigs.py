import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
from matplotlib import pyplot as plt

import numpy as np
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
    pts = interiorPoint(-A,-b,-c, niter=5, starting_point = (x,l,s))

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

interiorPath()
