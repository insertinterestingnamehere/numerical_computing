from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as colors
import matplotlib . pyplot as plt
import numpy as np

def visualizeSurface(colrs,centers,n):
#A has the signal values in the first row, and the center of each face in
#the second through fourth rows.

#ColorMat = [[1 1 1];eye(3);[1 .64 0];[1 1 0]];%Standard Rubik's Cube
#ColorMat = [[1 1 1];eye(3);[0 0 0];[1 1 0]];%Standard Rubik's Cube
    fig = plt.figure()
    ax = Axes3D(fig)
    for j in xrange(6*n**2):
        center = centers[:,j]
        color = colrs[:,j]
        for m in xrange(3): #%%This code should be improved, it's simply to fix roundoff
            if color[m] > 1:
                color[m] = 1
        d = 1./n
        if abs(center[0]) == 1:
            x = center[0]*np.ones(4)
            y = center[1] + np.array([d,d,-d,-d])
            z = center[2] + np.array([d,-d,-d,d])
        if abs(center[1]) == 1:
            y = center[1]*np.ones(4)
            x = center[0] + np.array([d,d,-d,-d])
            z = center[2] + np.array([d,-d,-d,d]) 
        if abs(center[2]) == 1:
            z = center[2]*np.ones(4)
            x = center[0] + np.array([d,d,-d,-d])
            y = center[1] + np.array([d,-d,-d,d])     
    #%fill3(x,y,z,ones(4,1)*color); MonoChromatic
        verts = [zip(x/2+.5, y/2+.5,z/2+.5)]
        tri=Poly3DCollection(verts)
        tri.set_color(colors.rgb2hex(color))
        tri.set_edgecolor('k')
        ax.add_collection3d(tri)
    plt.show()