def ImgToGraphCol(A):

    isize=A.shape[0]
    jsize=A.shape[1]
    nodes=[]
    for i in range(0,isize):
        for j in range(0,jsize):
            #print j
            if (i!=isize-1):
                nodes.append((int(i+j*isize),int((i+1)+j*A.shape[0]),diff(A[i,j,:],A[i+1,j,:])))
            if (j!=jsize-1):
                nodes.append((int(i+j*isize),int(i+(j+1)*A.shape[0]),diff(A[i,j,:],A[i,j+1,:])))
            if(i!=0):
                nodes.append((int(i+j*isize),int((i-1)+j*A.shape[0]),diff(A[i,j,:],A[i-1,j,:])))
            if(j!=0):    
                nodes.append((int(i+j*isize),int(i+(j-1)*A.shape[0]),diff(A[i,j,:],A[i,j-1,:]))) 
    return nodes

def diff(A,B):
    thesum=0
    for i in xrange(len(A)):
        thesum = thesum + abs(A[i]-B[i])
    return thesum

def ImgToGraph(A):

    isize=A.shape[0]
    jsize=A.shape[1]
    nodes=[]
    for i in range(0,isize):
        for j in range(0,jsize):
            #print j
            if (i!=isize-1):
                nodes.append((int(i+j*isize),int((i+1)+j*A.shape[0]),abs(A[i,j]-A[i+1,j])))
            if (j!=jsize-1):
                nodes.append((int(i+j*isize),int(i+(j+1)*A.shape[0]),abs(A[i,j]-A[i,j+1])))
            if(i!=0):
                nodes.append((int(i+j*isize),int((i-1)+j*A.shape[0]),abs(A[i,j]-A[i-1,j])))
            if(j!=0):    
                nodes.append((int(i+j*isize),int(i+(j-1)*A.shape[0]),abs(A[i,j]-A[i,j-1]))) 
    return nodes

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
        
def kruskalInt(nodes, edges,divs ):
    forest = DisjointSet()
    mst = []
    for n in nodes:
        forest.add( n )
    sz = len(nodes) - 1
    for e in sorted( edges, key=itemgetter( 2 )):
        n1, n2, _ = e
        t1 = forest.find(n1)
        t2 = forest.find(n2)
        if t1 != t2:
            mst.append(e)
            sz -= 1
            if sz == divs:
                return forest #mst

            forest.union(t1, t2)

n=200
div=7000
#X = sp.misc.imread('test.jpg').astype(float)
X = sp.misc.imread('Taylor.jpg').astype(float)
Xhat=(X.sum(2)/3)
Xr=sp.misc.imresize(Xhat,(n,n))
#img_color = plt.imread('dream.png')

#Xr = (img_color[:,:,0]+img_color[:,:,1]+img_color[:,:,2])/3.0
edges=ImgToGraph(Xr)
nodes=[]
for x in range(n**2):
    nodes.append(int(x))
forest=kruskalInt(nodes,edges,div)
for x in forest:
    forest[x]=forest.find(x)
edge=sp.array(forest.values())
data = Counter(edge)
ed=data.most_common(5)
edge.resize(n,n)
plt.subplot(221)
plt.gray()
plt.imshow(Xr)
plt.subplot(222)
plt.imshow(Xr[:,:]*(edge.T==ed[1][0]))
plt.subplot(223)
plt.imshow(Xr[:,:]*(edge.T==ed[2][0]))
plt.subplot(224)
plt.imshow(Xr[:,:]*(edge.T==ed[0][0]))
plt.gray()
plt.show()