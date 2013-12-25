import numpy as np
import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
from matplotlib import pyplot as plt
import scipy.optimize as opt

#the function with which to create the data and later fit it
def func(x,a,b,c):
    return a*np.exp(-b*x) + c

#create the exact and random data
x = np.linspace(0,4,50)
y = func(x,2.5,1.3,0.5)
yn = y + 0.2*np.random.normal(size=len(x));

#perform the fit
popt, pcov = opt.curve_fit(func,x,yn)

#graphing the fit with the data

yfit = func(x,popt[0],popt[1],popt[2])

plt.scatter(x,yn,marker='.')
plt.plot(x,yfit)
plt.savefig("curve_fit.pdf")
