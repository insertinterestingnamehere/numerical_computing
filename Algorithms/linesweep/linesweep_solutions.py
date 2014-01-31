import numpy as np
import scipy.spatial as st
from matplotlib import pyplot as plt
from math import sqrt
import heapq as hq
from edge_intersections import edge_intersections, inside

# optimized metric function for simplified linesweep
# Consider giving this one to them.
def metric(p, X):
    dif = (X - p)
    return np.sqrt((dif * dif).sum(axis=-1))

# simplified linesweep
def pymindist_simple(Y,metric):
    X = Y.take(Y[:,0].argsort(), axis=0)
    r = metric(X[0], X[1])
    low = 0
    for i in range(2,len(X)):
        while X[low,0] < X[i,0] - r:
            low+=1
        if low < i:
            r = min(r, np.min(metric(X[i],X[low:i])))
    return r

# full linesweep
def pymindist(Y):
    X = Y.take(Y[:,0].argsort(),axis=0)
    low = 0
    dim = X.shape[1]
    n = X.shape[0]
    r = 0.
    for i in xrange(dim):
        dif = X[0,i] - X[1,i]
        r += dif * dif
    r = sqrt(r)
    for i in xrange(2, n):
        while X[low,0] + r < X[i,0]:
            low += 1
        for k in xrange(low, i):
            proc = True
            d = 0.
            for j in xrange(1,dim):
                dif = abs(X[k,j] - X[i,j])
                if r < dif:
                    proc = False
                    break
                d += dif * dif
            if proc:
                dif = X[k,0] - X[i,0]
                r = min(r, sqrt(d + dif * dif))
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
