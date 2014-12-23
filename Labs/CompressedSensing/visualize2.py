import matplotlib . pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as colors
import numpy as np

def visualizeEarth(facetM,verM,C):
    fig = plt.figure()
    ax = Axes3D(fig)
    x=np.zeros(3)
    y=np.zeros(3)
    z=np.zeros(3)
    l=len(facetM)
    for n in xrange(l):
        x[0] = verM[facetM[n,0]-1,0]
        y[0] = verM[facetM[n,0]-1,1]
        z[0] = verM[facetM[n,0]-1,2]
        x[1] = verM[facetM[n,1]-1,0]
        y[1] = verM[facetM[n,1]-1,1]
        z[1] = verM[facetM[n,1]-1,2]
        x[2] = verM[facetM[n,2]-1,0]
        y[2] = verM[facetM[n,2]-1,1]
        z[2] = verM[facetM[n,2]-1,2]
        verts = [zip(x*.5+.5,y*.5+.5,z*.5+.5)]
        tri=Poly3DCollection(verts)

        tri.set_color(colors.rgb2hex(C[n]))
        tri.set_edgecolor('k')
        ax.add_collection3d(tri)
    plt.show()