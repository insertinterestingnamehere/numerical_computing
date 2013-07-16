import scipy as sp
from scipy import sparse as spar
from scipy . sparse import linalg as sparla
from scipy import linalg as la
import numpy as np
from scipy import eye
from math import sqrt
import matplotlib . pyplot as plt

def kruskal(A):
    size=A.shape
    minSpanTree=sp.zeros(size)
    nodesTree=sp.arange(size[0])
    D=sp.ones(size)*sp.arange(size[0])
    C=la.triu(A)
    Q=sp.concatenate([[C.flatten()],[D.T.flatten()],[D.flatten()]])
    W=Q[:,sp.nonzero(C.flatten())[0]]
    edges=W[:,W[0,:].argsort()]
    i=0
    j=0
    #while np.sum(nodesTree!=nodesTree[0])>0:
    while (j<(size[0]-1)):
        now=edges[:,i]
        i=i+1
        if nodesTree[now[1]]!=nodesTree[now[2]]:
            minSpanTree[now[1],now[2]]=now[0]
            nodesTree[nodesTree==nodesTree[now[2]]]=nodesTree[now[1]]
            j=j+1
    return minSpanTree+minSpanTree.T

def prims(A,start):
    size=A.shape
    minSpanTree=sp.zeros(size)
    nodesTree=sp.arange(size[0])
    D=sp.ones(size)*sp.arange(size[0])
    Q=sp.concatenate([[A.flatten()],[D.T.flatten()],[D.flatten()]])
    edges=Q[:,sp.nonzero(A.flatten())[0]]
    Set=[start]
    for iter in range(size[0]-1):
        yes=sp.array([False]*edges.shape[1])
        no=sp.array([False]*edges.shape[1])
        for x in Set:
            yes=yes+(edges[1,:]==x)
            no=no+(edges[2,:]==x)
        candates=yes*(no==False)
        tempEdges=edges[:,candates]
        now=tempEdges[:,tempEdges[0,:].argmin()]
        minSpanTree[now[1],now[2]]=now[0]
        Set=Set+[now[2]]
    return minSpanTree+minSpanTree.T


A=sp.array([[0,7,4,1,2],[7,0,7,6,0],[4,7,0,2,0],[1,6,2,0,3],[2,0,0,3,0]])

kruskal(A)
prims(A)

pickle.load(open("MSTdata.txt"))



from operator import itemgetter

class DisjointSet(dict):
    def add(self, item):
        self[item] = item
 
    def find(self, item):
        parent = self[item]
 
        while self[parent] != parent:
            parent = self[parent]

        self[item] = parent
        return parent
    
    def union(self, item1, item2):
        self[item2] = self[item1]
        
def kruskalInt( nodes, edges ):
    forest = DisjointSet()
    mst = []
    for n in nodes:
        forest.add( n )
    sz = len(nodes) - 1
    for e in sorted( edges, key=itemgetter( 2 ) ):
        n1, n2, _ = e
        t1 = forest.find(n1)
        t2 = forest.find(n2)
        if t1 != t2:
            mst.append(e)
            sz -= 1
            #forest.union(t1, t2)
            if sz == 0:
                return mst

            forest.union(t1, t2)
            
kruskalInt(['0','1','2','3','4'],S)