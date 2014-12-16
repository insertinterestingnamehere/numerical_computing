import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import numpy as np
from matplotlib import pyplot as plt


def mc_circle():
    np.random.seed(42)
    points = np.random.rand(2, 500).T
    points = 4*(points-.5)
    pointsNorm = np.hypot(points[:,0],points[:,1]) <= 1
    InCircle = points[pointsNorm]
    OutCircle = points[~pointsNorm]
    plt.plot(InCircle[:,0], InCircle[:,1], 'r.')
    plt.plot(OutCircle[:,0], OutCircle[:,1], 'b.')

    # Plot the circle
    theta = np.linspace(0, 2*np.pi, 50)
    plt.plot(np.cos(theta),np.sin(theta),'k')

    plt.axes().set_aspect('equal')
    plt.axis([-2, 2, -2, 2])
    
    plt.savefig("MC_Circle.pdf")
    plt.clf()
    
    
if __name__ == "__main__":
    mc_circle()