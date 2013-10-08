import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
from matplotlib import pyplot as plt

import numpy as np

def variance():
    # create a vector with large variance
    x = np.random.normal(size=50,loc=0,scale=5)
    
    # create a vector with small variance
    y = np.random.normal(size=50,loc=4.0,scale=.2)
    
    dom = np.arange(50)
    ax = plt.subplot(1, 2, 1)
    plt.plot(dom,x,'ro',dom,y,'bo')
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect((x1-x0)/(y1-y0))
    
    ax = plt.subplot(1, 2, 2)
    plt.plot(dom,y,'bo',dom,y-y.mean(),'go')
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect((x1-x0)/(y1-y0))
    
    plt.savefig('variance.pdf')
    plt.clf()

def correlation():
    # create random data, small correlation
    x1 = np.random.randn(100)
    y1 = np.random.randn(100)

    # create uncorrelated data (orthogonal vectors)
    x2 = np.random.rand(100)
    y2 = np.random.rand(100)
    y2 = y2 - ((x2*y2).sum())*x2/((x2*x2).sum())

    # create positively correlated data
    x3 = np.arange(100) + np.random.normal(size=100, loc=0, scale=.5)
    y3 = -np.arange(100) + np.random.normal(size=100, loc=0, scale=2)

    ax = plt.subplot(1, 3, 1)
    plt.scatter(x1, y1)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect((x1-x0)/(y1-y0))
    ax = plt.subplot(1, 3, 2)
    plt.scatter(x2, y2)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect((x1-x0)/(y1-y0))
    ax = plt.subplot(1, 3, 3)
    plt.scatter(x3, y3)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect((x1-x0)/(y1-y0))
    plt.savefig('correlation.pdf')
    plt.clf()

if __name__ == "__main__":
    variance()
    correlation()
