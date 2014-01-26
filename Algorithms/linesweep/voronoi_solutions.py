import numpy as np
import scipy.spatial as st
from matplotlib import pyplot as plt
from math import sqrt
import heapq as hq
#from markov_solutions import findpath
from edge_intersections import edge_intersections, inside

# farthest point problem

def farthest(pts, xlims, ylims, n):
    # there are a ton of ways to do this, this is a shorter one
    ins = lambda pt: inside(pt, xlims, ylims)
    V = st.Voronoi(pts)
    KD = st.cKDTree(pts)
    Q = [(KD.query(pt)[0],pt) for pt in V.vertices if ins(pt)]
    Q += [(KD.query(pt)[0],pt) for pt in edge_intersections(V, xlims, ylims)[0]]
    Q += [(KD.query(pt)[0],(x,y)) for x in xlims for y in ylims]
    return np.array([pair[1] for pair in hq.nlargest(n,Q)])

# obstacle navegation problem
# this problem is commented out
def make_adj(V, xlims, ylims, threshold):
    ins = lambda pt: inside(pt, xlims, ylims)
    size = V.vertices.shape[0]
    A = np.zeros((size,size),dtype=bool)
    for edge, centers in zip(V.ridge_vertices, V.ridge_points):
        if -1 not in edge:
            if threshold <= sqrt(np.sum((V.points[centers[0]]-V.points[centers[1]])**2))/2.:
                A[edge[0],edge[1]] = True
    A += A.T
    return A

def plot_path(V, i, j, xlims, ylims, threshold):
    A = make_adj(V, xlims, ylims, threshold)
    path = V.vertices[findpath(i,j,A)]
    st.voronoi_plot_2d(V)
    plt.plot(path[:,0], path[:,1])
    plt.show()

# triangulation of the unit squre problem

def triangulate(n):
    X = np.linspace(0,1,n)
    Y = X.copy()
    X,Y = np.meshgrid(X,Y,copy=False)
    A = np.column_stack((X.flat,Y.flat))
    D = st.Delaunay(A)
    plt.triplot(A[:,0],A[:,1],D.simplices.copy())
    plt.show()
