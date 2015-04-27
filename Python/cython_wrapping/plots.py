# Because the plot generated for this lab requires c installation,
# we use the laplace function from the Numpy lab which also generates the desired
# result.
import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

png_size = (800, 600)


def laplace(U, tol):
    new = U.copy()
    dif = tol
    while tol <= dif:
        new[1:-1,1:-1] = (U[:-2,1:-1] + U[2:,1:-1] + U[1:-1,:-2] + U[1:-1,2:])/4.0
        dif = np.max(np.absolute(U-new))
        U[:] = new

def cywrap_sol():
	resolution = 301
	U =  np.zeros((resolution, resolution))
	X = np.linspace(0, 1, resolution)
	U[0] = np.sin(2 * np.pi * X)
	U[-1] = -U[0]
	laplace(U, .000001)
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	u = np.linspace(0, 1, resolution)
	v = np.linspace(0, 1, resolution)
	u,v = np.meshgrid(u,v)

	ax.plot_surface(u, v, U, color='b')
	plt.savefig("solution.png", dpi=99)
	plt.clf()
	
if __name__ == "__main__":
	cywrap_sol()
