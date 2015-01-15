import numpy as np
import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
from matplotlib import pyplot as plt

def trapezoid_plot(f, xmin, xmax, n, name, res=1001):
    part = np.linspace(xmin, xmax, n+1)
    X = np.linspace(xmin, xmax, res)
    for i in xrange(part.size-1):
        plt.fill_between(part[i:i+2], f(part[i:i+2]))
    plt.plot(X, f(X), "r")
    plt.savefig(name)
    plt.cla()

if __name__ == "__main__":
    f = np.poly1d([-1,0,0,0,1])
    trapezoid_plot(f, 0, 1, 1, "Trapezoid.pdf")
    trapezoid_plot(f, 0, 1, 6, "TrapezoidComp.pdf")
