import scipy as sp
from scipy import sparse as spar
from scipy.sparse import linalg as sparla
from scipy import linalg as la
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import timeit

#sultion to problem 1
def nearestNNaive(points,x):
    l= len(points)
    r= sum ((x-points[0])**2)
    point=0
    for i in xrange (l):
        d= sum ((x-points[i])**2)
        if d<r:
            r=d
            point=i
    return r**(.5),point

#builds a kdtree
class Node: pass
 
def kdtree(points, depth=0):
 
    if len(points)==0:
        return None
 
    k = len(points[0])
    axis = depth % k
 
    points=points.take(points[:,axis].argsort(),axis=0)
    median = len(points) / 2
 
    # Create node and construct subtrees
    node = Node()
    node.location = points[median]
    node.left_child = kdtree(points[:median], depth + 1)
    node.right_child = kdtree(points[median + 1:], depth + 1)
    return node

#Helper function to KDstart. searches the kd-tree using recursion, Algortihm can problaly simplified.  
def KDsearch(node,point,best,bpoint,depth=0):
    if node==None:
        return best,bpoint
    
    k = len(node.location)
    axis = depth % k
    d=sum ((point-node.location)**2)
    if d<best:
        best=d
        bpoint=node.location[:]
    if point[axis]<node.location[axis]:
        best,bpoint=KDsearch(node.left_child,point,best,bpoint,depth+1)
        if point[axis]+best>=node.location[axis]:
           best,bpoint=KDsearch(node.right_child,point,best,bpoint,depth+1)
    else:
        best,bpoint=KDsearch(node.right_child,point,best,bpoint,depth+1)
        if point[axis]-best<=node.location[axis]:
           best,bpoint=KDsearch(node.left_child,point,best,bpoint,depth+1)
        
    return best,bpoint


#Starts the search of the KD-tree.
def KDstart(tree,point):
    best,bpoint=KDsearch(tree,point,sum ((point-tree.location)**2),tree.location)
    return best**(.5),bpoint

#timer function used to find the times for problems 3-5
def timeFun(f,*args,**kargs):
    pfunc = lambda: f(*args, **kargs)
    theTime=timeit.Timer(pfunc)
    return min(theTime.repeat(1,1))

#Graphs Problem 3. k-d search is a lot faster
numk=[]
times1=[]
times2=[]
k=4
for i in range(10):
    n=10000*(i+1)
    points=sp.rand(n,k)
    x=sp.rand(k)
    numk.append(n)
    tree=kdtree(points)
    times1.append(timeFun(KDstart,tree,x))
    times2.append(timeFun(nearestNNaive,points,x))

plt.plot(numk,np.array([times1,times2]).T)
#plt.savefig("4dTime.pdf")
plt.show()


#Grpahs problem 4. Both algorithms are about the same or Naive approach is slightly faster
numk=[]
times1=[]
times2=[]
k=20
for i in range(10):
    n=10000*(i+1)
    points=sp.rand(n,k)
    x=sp.rand(k)
    numk.append(n)
    tree=kdtree(points)
    times1.append(timeFun(KDstart,tree,x))
    times2.append(timeFun(nearestNNaive,points,x))

plt.plot(numk,np.array([times1,times2]).T)
#plt.savefig("20dTime.pdf")
plt.show()


#Graphs problem 5. Around dimension 15 the time spikes up 
numk=[]
times=[]
n=20000
for i in range(49):
    k=2+i
    points=sp.rand(n,k)
    x=sp.rand(k)
    numk.append(k)
    tree = KDTree(points)
    times.append(timeFun(tree.query,x))

plt.plot(numk,times)
plt.savefig("curseD.pdf")
plt.show()

