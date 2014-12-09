import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import timeit
from sklearn import neighbors
import solutions

def fourDTime():
    numk = []
    times1 = []
    times2 = []
    k = 4
    for i in xrange(1, 11):
        n = 10000 * i
        points = rand(n, k)
        x = rand(k)
        numk.append(n)
        tree = kdtree(points)
        times1.append(timeFun(KDstart, tree, x))
        times2.append(timeFun(nearestNNaive, points, x))
    plt.plot(numk, np.array([times1, times2]).T)
    plt.ylim((-.1, 1.))
    plt.savefig("fourDTime.pdf")
    plt.clf()
 

def twentyDTime():
    numk = []
    times1 = []
    times2 = []
    k = 20
    for i in range(1, 11):
        n = 10000 * i
        points = rand(n, k)
        x = rand(k)
        numk.append(n)
        tree = solutions.kdtree(points)
        times1.append(solutions.timeFun(solutions.KDstart, tree, x))
        times2.append(solutions.timeFun(solutions.nearestNNaive, points, x))

    plt.plot(numk, np.array([times1, times2]).T)
    plt.savefig("twentyDTime.pdf")
    plt.clf()


def curseD():
    numk = []
    times = []
    for i in xrange(49):
        k = 2 + i
        points = rand(20000, k)
        x = rand(k)
        numk.append(k)
        tree = KDTree(points)
        times.append(solutions.timeFun(tree.query, x))
    
    plt.plot(numk, times)
    plt.savefig("curseD.pdf")
    plt.clf()

if __name__ == "__main__":
    fourDTime()
    twentyDTime()
    curseD()
