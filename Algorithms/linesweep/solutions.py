import scipy as sp
import bintrees as bt
import pqueue as pq
import bisect as bs
from collections import deque
import itertools as it


#simplified version modified to work in arbitrary dimensions

def multidist(p0,p1):
    l=len(p0)
    return (sum([(p0[i]-p1[i])**2 for i in range(l)]))**(.5)

def test2(p,pt,r):
    for i in xrange(1,len(pt)):
        if r <= abs(p[i]-pt[i]):
            return False
    return True

def mindist18(X):
    ptlist=pq.PriorityQueue()
    n = len(X)
    for i in xrange(n):
        ptlist.add(tuple(X[i]),X[i,0])
    actives = []
    pt=ptlist.get()
    actives.append(pt)
    pt=ptlist.get()
    actives.append(pt)
    r = multidist(actives[0], actives[1])
    for i in xrange(n-2):
        pt=ptlist.get()
        actives[:]=[p for p in actives if ((pt[0]+r) > p[0])]
        for p in actives:
            if test2(p,pt,r):
                d=multidist(pt,p)
                if d<r:
                    r=d
        actives.append(pt)
    return r


#basic 2d implementation
def mindistRyan(X, width=5):
    ptlist=pq.PriorityQueue()
    for i in X:
        ptlist.add(tuple(i),i[0])
    actives = []
    r = width
    _bl = bs.bisect_left
    _br = bs.bisect_right
    _islice = it.islice
    pt1, pt2 = None, None
    while ptlist.size > 0:
        x, y = ptlist.get()
        _tmp = (y, x)
        
        actives[:] = [p for p in actives if (p[1] - x) <= r]
        
        for i in _islice(actives, _bl(actives, (y - r, x)), _br(actives, (y + r, x))):
            d = ((i[0]-y)**2 + (i[1]-x)**2)**.5
            if d < r:
                r = d
                pt1 = (x,y)
                pt2 = (i[1], i[0])
        actives.insert(_bl(actives, _tmp), _tmp)
    return (r, pt1, pt2)


#modified to do arbitrary dimensions
def ptorder(p,i):
    newpt=list(p)
    temp=newpt.pop(i)
    newpt.insert(0,temp)
    return tuple(newpt)

def revptorder(p,i):
    newpt=list(p)
    temp=newpt.pop(0)
    newpt.insert(i,temp)
    return tuple(newpt)

def rshift1(p,r):
    newp=list(p)
    newp[0]=p[0]-r
    newp[1]=p[1]+r
    return tuple(newp)

def rshift2(p,r):
    newp=list(p)
    newp[0]=p[0]+r
    return tuple(newp)

def test(p,pt,r):
    for i in xrange(2,len(pt)):
        if r <= abs(p[i]-pt[i]):
            return False
    return True

def multimindistRyan(X, width=5):
    ptlist=pq.PriorityQueue()
    for i in X:
        ptlist.add(tuple(i),i[0])
    actives = []
    r = width
    _bl = bs.bisect_left
    _br = bs.bisect_right
    _islice = it.islice
    while ptlist.size > 0:
        revpt = ptorder(ptlist.get(),1)
        
        actives[:] = [p for p in actives if (p[1] - revpt[1]) <= r]
        
        for p in _islice(actives, _bl(actives, rshift1(revpt,r)), _br(actives, rshift2(revpt,r))):
            if test(p,revpt,r):
                d = (sum([(p[i]-revpt[i])**2 for i in xrange(len(revpt))]))**(.5)
                if d < r:
                    r = d
        actives.insert(_bl(actives, revpt), revpt)
    return r


#version using binary trees

#uses the same helper functions as above

def mindist17(X):
    ptlist=pq.PriorityQueue()
    for i in xrange(len(X)):
        ptlist.add(tuple(X[i]),X[i,0])
    yactives = bt.FastAVLTree()
    xactives = deque()
    pt0=ptlist.get()
    pt1=ptlist.get()
    r=(sum([(pt0[i]-pt1[i])**2 for i in range(len(pt0))]))**(.5)
    yactives.insert(ptorder(pt0,1),True)
    yactives.insert(ptorder(pt1,1),True)
    xactives.append(pt0)
    xactives.append(pt1)
    for i in xrange(len(X)-2):
        pt=ptlist.get()
        revpt=ptorder(pt,1)
        removallist=[]
        while len(xactives)>0:
            if xactives[0][0]>=(pt[0]+r):
                removallist.append(ptorder(xactives[0],1))
                xactives.popleft()
            else:
                break
        yactives.delitems(removallist)
        block=yactives[rshift1(revpt,r):rshift2(revpt,r)]
        for p in block:
            if test(p,pt,r):
                d=(sum([(p[i]-revpt[i])**2 for i in range(len(pt))]))**(.5)
                if d<r:
                    r=d
        yactives.insert(revpt,True)
        xactives.append(pt)
    return r