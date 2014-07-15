import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def randomWalk():
    """Creates plot of symmetric one-D random lattice walk"""
    N = 1000        #length of random walk
    s = np.zeros(N)
    s[1:] = np.random.binomial(1, .5, size=(N-1,))*2-1 #coin flips
    s = pd.Series(s)
    s = s.cumsum()  #random walk
    s.plot()
    plt.ylim([-50,50])
    plt.savefig("randomWalk.pdf")
#randomWalk()

def biasedRandomWalk():
    """Create plots of biased random walk of different lengths."""
    N = 100        #length of random walk
    s1 = np.zeros(N)
    s1[1:] = np.random.binomial(1, .51, size=(N-1,))*2-1 #coin flips
    s1 = pd.Series(s1)
    s1 = s1.cumsum()  #random walk
    plt.subplot(211)
    s1.plot()

    N = 10000        #length of random walk
    s1 = np.zeros(N)
    s1[1:] = np.random.binomial(1, .51, size=(N-1,))*2-1 #coin flips
    s1 = pd.Series(s1)
    s1 = s1.cumsum()  #random walk
    plt.subplot(212)
    s1.plot()

    plt.savefig("biasedRandomWalk.pdf")
#biasedRandomWalk()

def dfPlot():
    """Plot columns of DataFrame against each other."""
    xvals = pd.Series(np.sqrt(np.arange(1000)))
    yvals = pd.Series(np.random.randn(1000).cumsum())
    df = pd.DataFrame({'xvals':xvals,'yvals':yvals}) #Put in in a dataframe
    df.plot(x='xvals',y='yvals') #Plot, specifying which column is to be used to x and y values.
    plt.savefig("dfPlot.pdf")
#dfPlot()

def histogram():
    """Creat histogram of columns in DataFrame."""
    col1 = pd.Series(np.random.randn(1000))   #normal distribution
    col2 = pd.Series(np.random.gamma(5, size=1000)) #gamma distribution 
    df = pd.DataFrame({'normal':col1, 'gamma':col2})
    df.hist()
    plt.savefig("histogram.pdf")
histogram()
