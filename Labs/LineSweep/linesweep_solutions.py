import numpy as np
import scipy.spatial as st
from matplotlib import pyplot as plt
from math import sqrt
import heapq as hq
from edge_intersections import edge_intersections, inside

# optimized metric function for simplified linesweep
# Consider giving this one to them.
def metric(p, X):
    # Finds distance between point 'p' and each of the rows of 'X'.
    # Works assuming 'p' is either 1-dimensional or a row vector.
    # 'X' can be a single 1-dimensional vector, a single row-vector,
    # or 2-dimensional array.
    dif = (X - p)
    return np.sqrt((dif * dif).sum(axis=-1))

# simplified linesweep
def pymindist_simple(Y, metric):
    """ Run the simple minimum distance algorithm explained in the lab.
    'Y' is the array of points. One point for each row.
    'metric' is a distance function."""
    # Sort by first coordinate.
    X = Y.take(Y[:,0].argsort(), axis=0)
    r = metric(X[0], X[1])
    # Use indices to track which points in the list are "active".
    low = 0
    for i in range(2, len(X)):
        # Update the 'low' index to reflect which points
        # still need further processing.
        while X[low,0] < X[i,0] - r:
            low += 1
        # If there really are any points to process,
        # update the minimum accordingly.
        if low < i:
            r = min(r, np.min(metric(X[i], X[low:i])))
    return r

# full linesweep
def pymindist(Y):
    """ Run the full minimum distance line sweep algorithm.
    'Y' is an array of points. One point for each row."""
    # Sort by first coordinate.
    X = Y.take(Y[:,0].argsort(), axis=0)
    # Use indices to track which points in the list are "active".
    low = 0
    dim = X.shape[1]
    n = X.shape[0]
    # Compute the starting distance.
    r = 0.
    for i in xrange(dim):
        dif = X[0,i] - X[1,i]
        r += dif * dif
    r = sqrt(r)
    # Process the rest of the points.
    for i in xrange(2, n):
        # Update the 'low' index to reflect which points
        # still need further processing.
        while X[low,0] + r < X[i,0]:
            low += 1
        # Process each point, rejecting it as soon as possible.
        for k in xrange(low, i):
            # Set a flag so the first coordinate is processed.
            # Don't process it at the beginning of the for-loop
            # since we already know those coordinates are close enough.
            proc = True
            # Start computing the distance.
            d = 0.
            for j in xrange(1, dim):
                # Compute absolute difference, then add in the
                # square of the difference if it is still in-bounds.
                dif = abs(X[k,j] - X[i,j])
                # Reject the point if it is already too far.
                if r < dif:
                    proc = False
                    break
                d += dif * dif
            # Finish processing the point if it hasn't been rejected yet.
            if proc:
                dif = X[k,0] - X[i,0]
                r = min(r, sqrt(d + dif * dif))
    return r

# farthest point problem
def farthest(pts, xlims, ylims, n):
    """ Find the 'n' points that lie farthest from the points given
    in the region bounded by 'xlims' and 'ylims'.
    'pts' is an array of points.
    'xlims' and 'ylims are tuples storing the maximum and minimum
    values to consider along the x and y axes."""
    # There are a ton of ways to do this, this is a shorter one.
    # The 'inside' function tests whether or not a point is on
    # the interior of the given square.
    ins = lambda pt: inside(pt, xlims, ylims)
    # Construct the Voronoi diagram.
    V = st.Voronoi(pts)
    # Construct the KD Tree.
    KD = st.cKDTree(pts)
    # Now we'll construct a list of tuples where the first
    # entry is the distance from a point to the nearest node
    # and the second entry is a tuple with the coordinates for the point.
    # Process the vertices of the Voronoi diagram.
    Q = [(KD.query(pt)[0], pt) for pt in V.vertices if ins(pt)]
    # Process the intersections of the edges of the
    # Voronoi diagram and the edges of the box.
    Q += [(KD.query(pt)[0], pt) for pt in edge_intersections(V, xlims, ylims)[0]]
    # Process the corners of the box.
    Q += [(KD.query(pt)[0], (x, y)) for x in xlims for y in ylims]
    # Return the 'n' points with farthest distance from the points
    # used to generate the Voronoi diagram.
    return np.array([pair[1] for pair in hq.nlargest(n, Q)])

# triangulation of the unit squre problem
def triangulate(n):
    """ Triangulate the square [0,1]x[0,1] using a grid with
    'n' equispaced points along each of its edges."""
    # Generate a grid of points.
    X = np.linspace(0, 1, n)
    Y = X.copy()
    X, Y = np.meshgrid(X, Y, copy=False)
    # Restructure the points generated so you can pass them
    # to the Delaunay class constructor.
    A = np.column_stack((X.flat, Y.flat))
    # Make a Delaunay triangulation.
    D = st.Delaunay(A)
    # Plot it.
    plt.triplot(A[:,0], A[:,1], D.simplices.copy())
    plt.show()
