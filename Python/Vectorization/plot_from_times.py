import numpy as np
import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
from matplotlib import pyplot as plt

times = np.load("times.npy")
ax = plt.subplot(1,1,1)
plt.plot(times[0], times[1], label="dist1")
plt.plot(times[0], times[2], label="dist2")
plt.plot(times[0], times[3], label="dist3")
plt.plot(times[0], times[4], label="dist4")
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, labels, loc="lower right")
plt.xlabel("Number of Points")
plt.ylabel("$\\log_{10}$ of Time")
plt.suptitle("Timings for Different Levels of Vectorization")
plt.savefig("distplot.pdf")
plt.cla()
