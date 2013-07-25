import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import numpy as np
import array_solutions
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

def lapace_figure():
    n = 100
    tol = .0001
    U = np.ones((n, n))
    U[:,[0, 1]]  = 100
    U[[0, 1]] = 0
    array_solutions.laplace(U, tol)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.linspace(0, 1, n)
    Y = X.copy()
    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X, Y, U, rstride=5)
    plt.savefig("laplace.pdf")
    
if __name__ == "__main__":
    lapace_figure()
