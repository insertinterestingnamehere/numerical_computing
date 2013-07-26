import numpy as np
import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
from matplotlib import pyplot as plt

#load .npy files
dot_results = np.load("dot_results.npy")
rowdot_results = np.load("rowdot_results.npy")

# plot dot product results
ax = plt.subplot(1,1,1)
p1, = plt.plot(dot_results[:,0], np.log(dot_results[:,1]), label="Python times")
p2, = plt.plot(dot_results[:,0], np.log(dot_results[:,2]), label="Cython times")
p3, = plt.plot(dot_results[:,0], np.log(dot_results[:,3]), label="Numpy with MKL times")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="lower right")
plt.xlabel("Array Size")
plt.ylabel("Natural Log of Running Time")
plt.savefig("dot.pdf")
ax.cla()

# plot rowdot results
p1, = plt.plot(rowdot_results[:,0], np.log(rowdot_results[:,1]), label="Python times")
p2, = plt.plot(rowdot_results[:,0], np.log(rowdot_results[:,2]), label="Cython times")
p3, = plt.plot(rowdot_results[:,0], np.log(rowdot_results[:,3]), label="Numpy with MKL times")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="lower right")
plt.xlabel("Number of Rows (all arrays had 3 columns)")
plt.ylabel("Natural Log of Running Time")
plt.savefig("rowdot.pdf")
ax.cla()
