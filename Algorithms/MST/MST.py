import numpy as np
from operator import itemgetter

def kruskal(nodes,A):
    group=createDict(nodes)
    sortedEdges=A.take(A[:,2].astype(int).argsort(),axis=0)
    s=len(nodes)
    i=0
    j=0
    edges=np.empty((s-1,3),dtype='|S2')
    while j<s-1:
        if group.find(sortedEdges[i,0])!=group.find(sortedEdges[i,1]):
            edges[j,:]=sortedEdges[i]
            group.union(group.find(sortedEdges[i,0]),group.find(sortedEdges[i,1]))
            j=j+1
        i=i+1
    return edges
            

def createDict(nodes):
    group=DisjointSet()
    s=len(nodes)
    for i in nodes:
        group.add(i)
    return group

#This is a solution to problem about disjoint sets that comes from http://programmingpraxis.com/

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

nodes = list( "ABCDEFG" )
edges = [ ("A", "B", 7), ("A", "D", 5),
          ("B", "C", 8), ("B", "D", 9), ("B", "E", 7),
      ("C", "E", 5),
      ("D", "E", 15), ("D", "F", 6),
      ("E", "F", 8), ("E", "G", 9),
      ("F", "G", 11)]

kruskal(nodes,np.array(edges))

def prims(nodes,edges):
    def change(t):
        for q in list(canid):
            if q[0]==t or q[1]==t:
                canid.remove(q)
            
        for q in list(m):
            if q[0]==t or q[1]==t:
                m.remove(q)
                canid.append(q)
        return
    m=list(edges)
    sol=[]
    canid=[]
    nadded=[]
    temp=min(m,key=itemgetter( 2 ))
    sol.append(temp)
    change(temp[0])
    change(temp[1])
    nadded.append(temp[0])
    nadded.append(temp[1])
    sz = len(nodes) - 2
    for i in xrange(sz):
        temp=min(canid,key=itemgetter( 2 ))
        sol.append(temp)
        if nadded.count(temp[0])==0:
            change(temp[0])
            nadded.append(temp[0])
        else:
            change(temp[1])
            nadded.append(temp[1])
    return sol

prims(nodes,edges)

def formChanger(oldData):
    newData=[]
    for i in oldData:
        newData.append((i[0],i[1],int(i[2])))
    return newData

q=np.load("MSTdata.npy")
edges=formChanger(q)
nodes = list( "ABCDEFGHIJKLMNOPQRSTUVWXYZabcd")

kruskal(nodes,np.array(edges))
prims(nodes,edges)