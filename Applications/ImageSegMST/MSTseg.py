from operator import itemgetter

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

# This is a modified version of the solution
# to the first problem in the MST lab.
def kruskalInt(edges, nodelist, div):
    removed = []
    nodes = {node:node for node in nodelist}
    remaining = len(nodes) - 1
    end = len(nodes) - div
    def track(node):
        temp = node
        while nodes[temp] is not temp:
            temp = nodes[temp]
        return temp
    for n1, n2, weight in sorted(edges, key=itemgetter(2)):
        root = track(n1)
        remove = track(n2)
        if root is not remove:
            removed.append(remove)
            remaining -=1
            if remaining == 0:
                return {node:track(node) for node in nodes}
            nodes[remove] = root

n=200
div=7000
#X = sp.misc.imread('test.jpg').astype(float)
X = sp.misc.imread('Taylor.jpg').astype(float)
Xhat = (X.sum(2) / 3)
Xr = sp.misc.imresize(Xhat, (n,n))
#img_color = plt.imread('dream.png')

#Xr = (img_color[:,:,0]+img_color[:,:,1]+img_color[:,:,2])/3.0
edges=ImgToGraph(Xr)
nodes=[]
for x in xrange(n**2):
    nodes.append(x)
forest=kruskalInt(edges, nodes, div)
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
