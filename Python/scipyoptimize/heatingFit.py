import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize as opt

#loading in and plotting the data
data = np.loadtxt("heating.txt")
#plt.scatter(data[:,0],data[:,1],marker='.',linewidths=0)
#plt.show()

#defining the function to which to fit to
T_a = 290
P = 59.34
def func(t, gamma, C, A):
    return T_a + P/gamma + A*np.exp(-gamma/C*t)

#plotting a guess together with the data
guess = [0.5,100,-100];
guessFunc = lambda t: func(t,guess[0],guess[1],guess[2])
guessData = guessFunc(data[:,0])

#plt.scatter(data[:,0],data[:,1],marker='.',linewidths=0)
#plt.plot(data[:,0],guessData)
#plt.show();

#the fit without a guess. On my machine it worked without the guess too
#popt, pcov = opt.curve_fit(func, data[:,0],data[:,1])

popt, pcov = opt.curve_fit(func, data[:,0],data[:,1],p0=guess)

#plotting the fit
fitFunc = lambda t: func(t,popt[0],popt[1],popt[2])
fitData = fitFunc(data[:,0])

plt.plot(data[:,0],fitData)
plt.scatter(data[:,0],data[:,1],marker='.',linewidths=0,color="black")
#plt.show()
plt.savefig("HeatingFit.pdf")
#print popt
#print pcov
