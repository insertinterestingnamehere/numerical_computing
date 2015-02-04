import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import numpy as np
from matplotlib import pyplot as plt
import scipy as sp

def dog_plots():
    # create data points
    x = np.linspace(0, 4, 100)
    y = 5000/4*x+50000
    zero=0*x
    
    plt.figure(num=1, figsize=(8, 3) )
    
    
    plt.subplot(1,2,1)
    plt.plot(x,y)
    plt.ylim(0,65000)
    plt.title("y-range 0-65,000")
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    
    plt.subplot(1,2,2)
    plt.plot(x,y)
    plt.ylim(49000,56000)
    plt.title("y-range 49,000-56,000")
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)

    plt.savefig("dog_plots.pdf")
    plt.close()
    
    
    
def heatmap_color():
    plt.figure(num=1, figsize=(10, 4) )
    n = 401
    x = np.linspace(-1,4,n)
    y = np.linspace(-4,1,n)
    X, Y = np.meshgrid(x,y)
    Z = np.sin(X)*np.sin(Y)
    
    plt.pcolormesh(X, Y, Z)
    plt.colorbar(ticks=[-1,0,.8])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("heatmap_color.png", bbox_inches='tight')
    plt.close()

def heatmap_gray():
    plt.figure(num=1, figsize=(10, 4) )
    n = 401
    x = np.linspace(-1,4,n)
    y = np.linspace(-4,1,n)
    X, Y = np.meshgrid(x,y)
    Z = np.sin(X)*np.sin(Y)
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.pcolormesh(X, Y, Z, cmap="gray")
    plt.colorbar(ticks=[-1,0,.8])
    plt.savefig("heatmap_gray.png", bbox_inches='tight')
    plt.close()
    
def healthcare_linscale():

    m = 2.07
    s = 0.63

    num_samples = 10000
    samples = []

    for i in xrange(num_samples):
        samples.append(sp.random.lognormal(m, s)) 

    sp_samples = sp.array(samples)

    #plt.figure(num=1, figsize=(10, 4) )

    plt.hist(sp.exp(sp_samples), 100)
	
    plt.savefig("healthcare_linscale.pdf")
    plt.close()
    
def healthcare_logscale():
    m = 2.07
    s = 0.63

    num_samples = 10000
    samples = []

    for i in xrange(num_samples):
        samples.append(sp.random.lognormal(m, s)) 

    sp_samples = sp.array(samples)

    plt.hist(sp_samples, 100)
	
    plt.savefig("healthcare_logscale.pdf")
    plt.close()
	 
def simplify_plot():
    m = 2.07
    s = 0.63

    num_samples = 10000
    samples = []

    for i in xrange(num_samples):
        samples.append(sp.random.lognormal(m, s)) 

    sp_samples = sp.array(samples)
    
    # Get rid of lines
    plt.hist(sp_samples, 100, histtype='stepfilled')
    
    # get current axis instance
    axis = plt.gca()
    # hide top and right spines
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    
    # Only show ticks on the left and bottom spines
    axis.yaxis.set_ticks_position('left')
    axis.xaxis.set_ticks_position('bottom')
    
    # Use fewer axis ticks
    plt.xlim(0,70)
    plt.xticks(np.arange(0, 71, 35))
    
    plt.ylim(0, 800)
    plt.yticks(np.arange(0, 801, 200))
    
    plt.savefig("simplify.pdf")
    plt.close()
    

    
	 
def log_plots():
    # create data points
    x = np.linspace(1, 10, 100)
    yvalues = [x, np.log(x), np.power(np.e, x), np.power(x, 3)]
    
    # create labels
    labels = ["y = x", "y = log(x)", "y = e^x", "y = x^3"]
    
    # plot lin-lin
    plt.subplot(2,2,1)
    for y, string in zip(yvalues, labels):
        plt.plot(x, y, label=string)
    plt.ylim(0,10)
    plt.title("lin-lin plot")
    
    # plot lin-log
    plt.subplot(2,2,2)
    for y, string in zip(yvalues, labels):
        plt.semilogx(x, y, label=string)
    plt.ylim(0,10)
    plt.title("lin-log plot")
    
    # plot log-lin
    plt.subplot(2,2,3)
    for y, string in zip(yvalues, labels):
        plt.semilogy(x, y, label=string)
    plt.title("log-lin plot")
    
    # plot log-log
    plt.subplot(2,2,4)
    for y, string in zip(yvalues, labels):
        plt.loglog(x, y, label=string)
    plt.title("log-log plot")  
    
    plt.savefig("log_plots.pdf")
    plt.close()
    
if __name__ == "__main__":
    dog_plots()
    heatmap_color()
    heatmap_gray()
    healthcare_linscale()
    healthcare_logscale()
    simplify_plot()
    log_plots()
    
