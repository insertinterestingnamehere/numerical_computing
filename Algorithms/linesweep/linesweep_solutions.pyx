from numpy cimport ndarray as ar
cimport cython
import numpy as np
import scipy.spatial as st
from matplotlib import pyplot as plt
from libc.math cimport fabs, sqrt
import heapq as hq
from edge_intersections import edge_intersections, inside

# optimized metric function for simplified linesweep
# Consider giving this one to them.
def metric(p, X):
    dif = (X - p)
    return np.sqrt((dif * dif).sum(axis=-1))

# simplified linesweep
def cymindist_simple(Y, metric):
    X = Y.take(Y[:,0].argsort(), axis=0)
    r = metric(X[0], X[1])
    cdef int low=0
    cdef int i = 0
    for i in range(2, len(X)):
        while X[low,0] < X[i,0] - r:
            low += 1
        if low < i:
            r = min(r, np.min(metric(X[i], X[low:i])))
    return r

# full linesweep
@cython.boundscheck(False)
@cython.wraparound(False)
def mindist(Y):
    cdef np.ndarray[double,ndim=2] X = Y.take(Y[:,0].argsort(), axis=0)
    cdef double d, dif, r = 0.
    cdef int low=0, dim=X.shape[1], n = X.shape[0], i, j, k, proc
    for i in range(dim):
        r+= (X[0,i] - X[1,i])**2
    r = sqrt(r)
    for i in range(2, n):
        while X[low,0] < X[i,0] - r:
            low += 1
        for k in range(low, i):
            proc = 1
            d = 0.
            for j in range(1, dim):
                dif = fabs(X[k,j] - X[i,j])
                if r < dif:
                    proc = 0
                    break
                d += dif**2
            if proc:
                r = min(r, sqrt(d + (X[k,0] - X[i,0])**2))
    return r

# farthest point problem
def farthest(pts, xlims, ylims, n):
    # there are a ton of ways to do this, this is a shorter one
    ins = lambda pt: inside(pt, xlims, ylims)
    V = st.Voronoi(pts)
    KD = st.cKDTree(pts)
    Q = [(KD.query(pt)[0], pt) for pt in V.vertices if ins(pt)]
    Q += [(KD.query(pt)[0], pt) for pt in edge_intersections(V, xlims, ylims)[0]]
    Q += [(KD.query(pt)[0], (x, y)) for x in xlims for y in ylims]
    return np.array([pair[1] for pair in hq.nlargest(n, Q)])

# triangulation of the unit squre problem
def triangulate(n):
    X = np.linspace(0, 1, n)
    Y = X.copy()
    X, Y = np.meshgrid(X, Y, copy=False)
    A = np.column_stack((X.flat, Y.flat))
    D = st.Delaunay(A)
    plt.triplot(A[:,0], A[:,1], D.simplices.copy())
    plt.show()
