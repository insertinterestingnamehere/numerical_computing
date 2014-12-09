import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as colors

def visualize3d(F, C):
    fig = plt.figure()
    ax = Axes3D(fig)
    l = len(F)
    for n in xrange(l):
        verts = [zip(F[n, 0]*.5+.5, F[n, 1]*.5+.5, F[n, 2]*.5+.5)]
        tri = Poly3DCollection(verts)
        tri.set_color(colors.rgb2hex(C[n]))
        tri.set_edgecolor('k')
        ax.add_collection3d(tri)
    plt.show()

from matplotlib.collections import PolyCollection
def visualize2d(verts, C):
    fig, ax = plt.subplots()
    coll = PolyCollection(verts, facecolors=C)
    ax.add_collection(coll)
    ax.autoscale_view()
    plt.show()

def showcolor(b):
	plt.imshow(np.array([[b]]))
	plt.show()