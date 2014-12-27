import numpy as np
from numpy.random import randint
from math import sin, cos
from operator import itemgetter
from matplotlib import pyplot as plt

def randsum(size, number):
    tot = 0.
    A = np.empty(size, dtype=np.float32)
    for i in xrange(number):
        A[:] = rand(size).astype(np.float32)
        tot += A.sum() # basically just sum them in groups instead of one value at a time
    return tot

def der(f, x, h):
    return (f(x+h) - f(x)) / h

def dertest(x):
    val = cos(x)
    approxs = [abs(der(sin, x, 10**i)-val) for i in xrange(-1, -16, -1)] # errors for different pows of 10
    min_index, min_val = min(enumerate(approxs), key=itemgetter(1)) # get index of min
    return -1 - min_index # returns optimal power of 10

def lnseries(n):
    poly = np.poly1d([(-1.)**(i+1)/i for i in xrange(1,n)][::-1]) # define series
    X = np.linspace(-5, 30, 17501) # high resolution x values
    Y = np.log(10**(-X) + 1) / 10**(-X) # normal computation
    plt.plot(X, Y) # make first plot
    X = np.linspace(1, 30, 291) # lower resolution for second plot
    Y = poly(10**(-X)) # Compute series values
    plt.plot(X, Y) # plot series
    plt.show() # display results

def find_triple(start, num):
	# This is horribly terribly slow in python.
	# I've included a cython version as weel.
    temp = 0.
    for i in xrange(num):
        for j in xrange(i, num):
            temp = (start+i)**2 + (start+j)**2
            temp = sqrt(temp)
            if int(temp+1)**2 - (start+i)**2 - (start+j)**2 == 0 : # only integer computations in checking
                print start+i, start+j, int(temp+1)
            if int(temp)**2 - (start+i)**2 - (start+j)**2 == 0: # checks both of the nearest integers
                print start+i, start+j, int(temp)

def patriot_sim():
    # double precision is close enough in this case
    dif = (.1 - 209715/2097152.) * 3600000
    print "Off by ", dif, " seconds."
    dist = 1676 * dif
    print "Missile location off by ", dist, " meters."
