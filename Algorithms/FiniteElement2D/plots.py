import numpy as np
import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mayavi import mlab as ml
from scipy.misc import imsave

def triangles(n):
    """ Generate the indices of the triangles for a triangular mesh
    on a square grid of points.
    'n' is expected to be the number of nodes on each edge. """
    # Make the indices for a single row.
    row = np.empty((2 * (n - 1), 3), dtype=np.int32)
    row[::2,0] = row[1::2,0] = row[::2,1] = np.arange(n-1)
    row[1::2,0] += 1
    row[::2,1] += n
    row[1::2,1] = row[::2,1]
    row[::2,2] = row[1::2,2] = row[1::2,0]
    row[1::2,2] += n
    # Now use broadcasting to make the indices for the square.
    return (row + np.arange(0, n * (n-1), n)[:,None,None]).reshape((-1,3))

def plot_square_triangulation(filename):
    n=5
    x = np.linspace(0, 1, n)
    x, y = map(np.ravel, np.meshgrid(x, x))
    t = triangles(n)
    plt.triplot(x, y, t, color='b')
    plt.scatter(x, y, color='b')
    plt.savefig(filename)
    plt.clf()

def plot_basis_function(filename):
    n=5
    x = np.linspace(0, 1, n)
    x, y = np.meshgrid(x, x)
    t = triangles(n)
    vals = np.zeros(x.size)
    vals[n**2 // 2] = 1
    ml.figure(size=(500,500))
    ml.triangular_mesh(x.ravel(), y.ravel(), vals, t)
    ml.pitch(-3.5)
    ml.gcf().scene.camera.position = ml.gcf().scene.camera.position / 1.2
    ml.gcf().scene.save_png(filename)
    ml.clf()

if __name__ == '__main__':
    plot_square_triangulation('square_triangulation.pdf')
    plot_basis_function('hat_function.png')
