import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import numpy as np
from matplotlib import pyplot as plt
from math import sqrt

def dogleg():
    # on the first segment
    rad = 1
    plt.gcf()
    circle1=plt.Circle((0,0),rad,color='r', fill=False, ls='dashed')
    fig = plt.gcf()
    fig.gca().add_artist(circle1)
    plt.xlim(-3, 5)
    plt.ylim(-4,4)
    d1 = np.linspace(0,1,2)
    plt.plot(d1,-d1, 'k')
    d2 = np.linspace(1,3.5,2)
    plt.plot(d2,-1.3+.3*d2, 'k')
    plt.plot(.5*sqrt(2), -.5*sqrt(2), 'o')
    plt.xticks([])
    plt.yticks([])
    plt.savefig("dogleg1.pdf")
    plt.clf()

    # on the second segment
    rad = 1.8
    plt.gcf()
    circle1=plt.Circle((0,0),rad,color='r', fill=False,ls='dashed')
    fig = plt.gcf()
    fig.gca().add_artist(circle1)
    plt.xlim(-3, 5)
    plt.ylim(-4,4)
    d1 = np.linspace(0,1,2)
    plt.plot(d1,-d1, 'k')
    d2 = np.linspace(1,3.5,2)
    plt.plot(d2,-1.3+.3*d2, 'k')
    plt.plot(1.6028, -1.3+.3*1.6028, 'o')
    plt.xticks([])
    plt.yticks([])
    plt.savefig("dogleg2.pdf")
    plt.clf()

    # on the final point
    rad = 3.8
    plt.gcf()
    circle1=plt.Circle((0,0),rad,color='r', fill=False,ls='dashed')
    fig = plt.gcf()
    fig.gca().add_artist(circle1)
    plt.xlim(-5, 5)
    plt.ylim(-5,5)
    d1 = np.linspace(0,1,2)
    plt.plot(d1,-d1, 'k')
    d2 = np.linspace(1,3.5,2)
    plt.plot(d2,-1.3+.3*d2, 'k')
    plt.plot(3.5,-1.3+.3*3.5, 'o')
    plt.xticks([])
    plt.yticks([])
    plt.savefig("dogleg3.pdf")
    plt.clf()

if __name__ == "__main__":
    dogleg()
