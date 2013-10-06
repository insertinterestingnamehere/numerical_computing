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
    plt.subplot(121)
    plt.plot(dom,x,'ro',dom,y,'bo')
    
    plt.subplot(122)
    plt.plot(dom,y,'bo',dom,y-y.mean(),'go')
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
    x3 = np.arange(100) + np.random.normal(size=100,loc=0,scale=.5)
    y3 = -np.arange(100) + np.random.normal(size=100,loc=0,scale=2)

    plt.subplot(131)
    plt.scatter(x1,y1)
    plt.subplot(132)
    plt.scatter(x2,y2)
    plt.subplot(133)
    plt.scatter(x3,y3)
    plt.savefig('correlation.pdf')
    plt.clf()
    
variance()
correlation()
