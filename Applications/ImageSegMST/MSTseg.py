from operator import itemgetter

#makes a list of the form (node,node, weight) where each pixel is a node and has about four edges.
#weight is defined as the sum of the absolute difference of each color value. For color pictures
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

#helper function
def diff(A,B):
    thesum=0
    for i in xrange(len(A)):
        thesum = thesum + abs(A[i]-B[i])
    return thesum

#Smae as above except weight is defined as difference between pixel values. For B&W pictures
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

#segments as image and plots it. n is the size of the image (you cann do up to 500 rather where)
#it does not have to be square
#div is the number of divisions and I have to guess and check to find out what a good number is
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
#the next 2 lines find the 5 trees with the most nodes
data = Counter(edge)
ed=data.most_common(5)
edge.resize(n,n)
#graphs the 3 bigest segments.
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
