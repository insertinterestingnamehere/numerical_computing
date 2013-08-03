import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

from matplotlib import pyplot as plt
import numpy as np
import subprocess

data = np.load("cache_times.npz")

def cache_lines():
    times = data['cache_line']
    plt.plot(np.log2(times[0]), times[1], marker='.')
    plt.ylabel("Runtime (s)")
    plt.xlabel("Step sizes (as powers of 2)")
    
    plt.annotate("Cache line size",
                 xy=(np.log2(times[0,4]), times[1,4]),
                 xytext=(2, .03),
                 arrowprops=dict(facecolor='black', shrink=.05))
    plt.savefig("cache_line.pdf")
    
def cache_sizes():
    plt.clf()
    times = data['cache_size']
    plt.plot(np.log2(times[0]), times[1], marker='.')
    plt.ylabel("Runtime (s)")
    plt.xlabel("Cache size (as powers of 2)")
    
    arrowprops = dict(facecolor='black', shrink=.05)
    plt.annotate("L1 cache limit",
                 xy=(np.log2(times[0,5]), times[1,5]),
                 xytext=(1, .25), arrowprops=arrowprops)
    plt.annotate("L2 cache limit",
                 xy=(np.log2(times[0,8]), times[1,8]),
                 xytext=(4, .45), arrowprops=arrowprops)
    plt.annotate("L3 cache limit",
                 xy=(np.log2(times[0,12]), times[1,12]),
                 xytext=(6, .6), arrowprops=arrowprops)
    plt.annotate("RAM",
                 xy=(np.log2(times[0,-1]), times[1,-1]),
                 xytext=(10, .9), arrowprops=arrowprops)
    plt.savefig("cache_size.pdf")
    
    
if __name__ == "__main__":
    cache_lines()
    cache_sizes()