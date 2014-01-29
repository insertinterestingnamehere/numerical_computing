from math import sqrt
import numpy as np
import scipy.spatial as st

def inside(pt, xlims, ylims, tol=1E-12):
    if xlims[0]-tol <= pt[0] and pt[0] <= xlims[1]+tol:
        if ylims[0]-tol <= pt[1] and pt[1] <= ylims[1]+tol:
            return True
    return False

def edge_intersections(V, xlims, ylims, tol=1E-12):
    """
    Finds intersections of edges with the boundaries of a rectangular region bounded by xlims and ylims.
    Returns an array of the coordinates of the intersections and two lists of pairs of indices.
    The first list of indices represents edges formed by one point inside the region and an intersection with the edge.
    The second list of indices represents edges formed between two intersections of the Voronoi diagram.
    This second case arises when the region does not contain any of the vertices corresponding to an edge it intersects.
    In the first list of indices, the first index in each pair corresponds to a vertex in V.vertices and the second index corresponds to one of the points in the array of edge-intersections.
    In the second list of indieces, both indices correspond to points in the new array of edge-intersections.
    V is a Voronoi diagram object.
    The bounds xlim and ylim are expected to be tuples of a minimum and a maximum value for the x and y coordinate respectively.
    """
    boundpts = []
    newedges = []
    outedges = []
    cent = V.points.mean(axis=0)
    ins = lambda pt: inside(pt, xlims, ylims, tol=tol)
    for cpts, edge in zip(V.ridge_points, V.ridge_vertices):
        # track number of intersections with boundaries for each edge
        oldlen = len(boundpts)
        # if the edge is infinite:
        if -1 not in edge:
            pt0 = V.vertices[edge[0]]
            pt1 = V.vertices[edge[1]]
            if not (ins(pt0) and ins(pt1)):
                i = None
                if ins(pt0):
                    i = edge[0]
                elif ins(pt1):
                    i = edge[1]
                # left and right sides
                # avoid zero division
                dif = pt1[0] - pt0[0]
                if dif != 0.:
                    for x in xlims:
                        c = (x - pt0[0]) / dif
                        # if 0<=c<=1 then this edge intersects the line between pt0 and pt1
                        newpt = (1.-c)*pt0 + c*pt1
                        if 0. <= c and c <= 1. and ins(newpt):
                            if i is not None:
                                newedges.append([i,len(boundpts)])
                            boundpts.append(newpt)
                # top and bottom
                dif = pt1[1] - pt0[1]
                if dif != 0.:
                    for y in ylims:
                        c = (y - pt0[1]) / dif
                        newpt = (1.-c)*pt0 + c*pt1
                        if 0. <= c and c <= 1. and ins(newpt):
                            if i is not None:
                                newedges.append([i,len(boundpts)])
                            boundpts.append(newpt)
        # if the edge is finite
        else:
            # index of finite endpoint of infinite edge
            i = [i for i in edge if i>=0][0]
            # get the vertex corresponding to i
            pt = V.vertices[i]
            # direction between the centers of cells separated by the infinite edge
            t = V.points[cpts[1]] - V.points[cpts[0]]
            # normalizing t
            t /= sqrt(t[0]**2 + t[1]**2)
            # normal vector to t
            n = np.array([-t[1], t[0]])
            # we need to find which direction is "out"
            # so we'll use the midpoint and the center
            mid = V.points[cpts].mean(axis=0)
            # d is the modified version of n in the proper direction
            d = np.sign(np.dot(mid - cent, n)) * n
            # find intersections with left and right sides
            if d[0] != 0.:
                c = (xlims[0] - pt[0]) / d[0]
                # only add it in if it runs in the right direction
                if c > 0:
                    # compute the new point
                    newpt = pt + c * d
                    # only add it in if it is in the domain
                    if ylims[0] <= newpt[1] and newpt[1] <= ylims[1]:
                        newedges.append([i,len(boundpts)])
                        boundpts.append(newpt)
                # now the same thing for the other side
                c = (xlims[1] - pt[0]) / d[0]
                if c > 0:
                    newpt = pt + c * d
                    if ylims[0] <= newpt[1] and newpt[1] <= ylims[1]:
                        newedges.append([i,len(boundpts)])
                        boundpts.append(newpt)
            # now the same thing for the top and bottom
            if d[1] != 0.:
                c = (ylims[0] - pt[1]) / d[1]
                if c > 0:
                    newpt = pt + c * d
                    if xlims[0] <= newpt[0] and newpt[0] <= xlims[1]:
                        newedges.append([i,len(boundpts)])
                        boundpts.append(newpt)
                c = (ylims[1] - pt[1]) / d[1]
                if c > 0:
                    newpt = pt + c * d
                    if xlims[0] <= newpt[0] and newpt[0] <= xlims[1]:
                        newedges.append([i,len(boundpts)])
                        boundpts.append(newpt)
        if len(boundpts) - oldlen == 2:
            # Account for the case that neither vertex is in the region
            # but there are still intersections.
            outedges.append([len(boundpts)-2, len(boundpts)-1])
    return np.array(boundpts), newedges, outedges
