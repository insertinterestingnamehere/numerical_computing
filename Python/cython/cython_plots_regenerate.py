import numpy as np
from numpy.random import rand
import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
from matplotlib import pyplot as plt
from timeit import timeit as ti
from os import system

system("dotsetup.py build_ext --inplace")
system("rowdotsetup.py build_ext --inplace")

from dot import cydot
from rowdot import cyrowdot

def pydot(A, B):
    tot = 0.
    for i in xrange(A.size):
        tot += A[i] * B[i]
    return tot

def pyrowdot(A):
    B = np.empty((A.shape[0], A.shape[0]))
    for i in xrange(A.shape[0]):
        for j in xrange(i):
            temp = pydot(A[i], A[j])
            B[i,j] = temp
        B[i,i] = pydot(A[i],A[i])
    for i in xrange(A.shape[0]):
        for j in xrange(i+1,A.shape[0]):
            B[i,j] = B[j,i]
    return B

dot_sizes = np.array([100, 500, 1000, 5000, 10000,
                      50000, 100000, 500000, 1000000,
                      5000000, 10000000, 15000000, 20000000,
                      25000000])

rowdot_sizes = np.array([10, 50, 100, 200, 300, 400, 500,
                         600, 700, 800, 900, 1000, 1100,
                         1200, 1300, 1400, 1500, 1600])

dot_times = np.empty((len(dot_sizes),3))

rowdot_times = np.empty((len(rowdot_sizes),3))

for n, times in zip(dot_sizes, dot_times):
    A = rand(n)
    B = rand(n)
    times[0] = ti("pydot(A, B)", setup="from __main__ import A, B, pydot", number=5) / 5.
    times[1] = ti("cydot(A, B)", setup="from __main__ import A, B, cydot", number=500) / 500.
    times[2] = ti("A.dot(B)", setup="from __main__ import A, B", number = 500) / 500.

for n, times in zip(rowdot_sizes, rowdot_times):
    A = rand(n, 3)
    times[0] = ti("pyrowdot(A)", setup="from __main__ import A, pyrowdot", number=1)
    times[1] = ti("cyrowdot(A)", setup="from __main__ import A, cyrowdot", number=10) / 10.
    times[2] = ti("A.dot(A.T)", setup="from __main__ import A", number=10) / 10.

# plot dot product results
ax = plt.subplot(1,1,1)
p1, = plt.plot(dot_sizes, np.log(dot_times[:,0]), label="Python times")
p2, = plt.plot(dot_sizes, np.log(dot_times[:,1]), label="Cython times")
p3, = plt.plot(dot_sizes, np.log(dot_times[:,2]), label="Numpy with MKL times")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="lower right")
plt.xlabel("Array Size")
plt.ylabel("Natural Log of Running Time")
plt.savefig("dot.pdf")
ax.cla()

# plot rowdot results
p1, = plt.plot(rowdot_sizes, np.log(rowdot_times[:,0]), label="Python times")
p2, = plt.plot(rowdot_sizes, np.log(rowdot_times[:,1]), label="Cython times")
p3, = plt.plot(rowdot_sizes, np.log(rowdot_times[:,2]), label="Numpy with MKL times")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="lower right")
plt.xlabel("Number of Rows (all arrays had 3 columns)")
plt.ylabel("Natural Log of Running Time")
plt.savefig("rowdot.pdf")
ax.cla()

#save data for dot
dot_results = np.empty((dot_sizes.size, 4))
dot_results[:,0] = dot_sizes
dot_results[:,1:] = dot_times
np.save("dot_results.npy", dot_results)

#save data for rowdot
rowdot_results = np.empty((rowdot_sizes.size, 4))
rowdot_results[:,0] = rowdot_sizes
rowdot_results[:,1:] = rowdot_times
np.save("rowdot_results.npy", rowdot_results)
