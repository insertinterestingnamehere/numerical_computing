import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
from matplotlib import pyplot as plt

import numpy as np

def feasibleRegion():
    dom = np.linspace(-1,5, 1000)
    y1 = -10./4*dom+10
    y2 = -4./10*dom+4
    y3 = -1*dom+5
    ymin = np.vstack([y1,y2,y3]).min(axis=0)
    plt.plot(dom, y1, 'b')
    plt.plot(dom, y2, 'b')
    plt.plot(dom, y3, 'b')
    plt.vlines(0, -1, 6, 'b')
    plt.plot(dom, 0*dom, 'b')
    plt.fill_between(dom, ymin, where=(dom>=0)*(dom<=4.), color = 'g', alpha=.2)
    plt.text(.5, 1.5, 'Feasible Polytope', fontsize=16)
    plt.ylim([-1, 5])
    plt.savefig('feasiblePolytope.pdf')
    plt.clf()
feasibleRegion()
