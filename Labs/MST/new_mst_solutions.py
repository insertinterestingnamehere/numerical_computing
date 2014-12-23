import numpy as np
import networkx as nx
import scipy.ndimage
from scipy import linalg as la
from operator import itemgetter
from collections import counter
from matplotlib import pyplot as plt

def make_edges(n):
    A = la.triu(np.random.randint(1,50,(n,n))*(np.random.rand(n,n)>.5))
    S = []
    for index, x in np.ndenumerate(A):
        if x != 0:
            S.append((str(index[0]), str(index[1]), x))
    return S

def formChanger(oldData):
    newData = []
    for i in oldData:
        newData.append((i[0],i[1],int(i[2])))
    return newData

# Problem 1

def kruskal(edges):
	# Empty list of edges for MST
    tree = []
    # Dictionary that points each node towards its root, initially itself
    nodes = {node:node for node in ({edge[0] for edge in edges} | {edge[1] for edge in edges})}
    # Set number of nodes to be processed to n-1
    remaining = len(nodes)-1
    # Find the root of the given node
    def track(node):
    	# Node whose root we are finding
        temp = node
        # While temp does not point to itself in the dictionary
        while nodes[temp] is not temp:
        	# Update temp to be the node it currently points to in nodes
            temp = nodes[temp]
        return temp
    for n1, n2, weight in sorted(edges, key=itemgetter(2)):
    	# Root node of n1
        root = track(n1)
        # Root node of n2
        remove = track(n2)
        if root is not remove:
        	# Add the edge to the tree
            tree.append((n1, n2, weight))
            # Lower remaining by 1
            remaining -= 1
            if remaining == 0:
                return tree
            # Change the value associated with remove to root
            nodes[remove] = root

# Problem 2

oldData = np.load('MSTdata.npy')
data = formChanger(oldData)
# Timing for kruskal(data): 338 microseconds per loop
G = nx.Graph()
for i in data:
	G.add_edge(i[0], i[1], weight=int(i[2]))
# Timing for nx.minimum_spanning_tree(G): 2.4 milliseconds per loop

# Problem 3
def convert(filename):
	picture = scipy.ndimage.imread(filename)
    A = picture[:,:,0]
    edges = []
    a = A.shape
    for index, x in np.ndenumerate(A):
        i = index[0]
        j = index[1]
        # Avoid the pixels on the edges
        if i < a[0]-1 and j < a[1]-1:
            # Only do the i+1 and j+1 cases since it's bidirectional
            edges.append((A[i,j], A[i,j+1], abs(A[i,j]-A[i,j+1])))
            edges.append((A[i+1,j], A[i,j], abs(A[i+1,j]-A[i,j])))
    return edges

# Problem 4
def modifiedkruskal(edges, div):
	# Create dictionary that points each node towards its root, initially itself
    nodes = {node:node for node in ({edge[0] for edge in edges} | {edge[1] for edge in edges})}
    # Set number of nodes to be processed to n-div
    end = len(nodes)-div
    # Tracking function- same as in original Kruskal
    def track(node):
        temp = node
        while nodes[temp] is not temp:
            temp = nodes[temp]
        	return temp
    for n1, n2, weight in sorted(edges, key=itemgetter(2)):
        root = track(n1)
        remove = track(n2)
        if root is not remove:
            end -=1
            if end == 0:
            	# Makes sure you get the right number of divisions
                nodes[remove] = root
                # Return dict with nodes as keys and their roots as values
                return {node:track(node) for node in nodes}
            # Change the value associated with remove to root
            nodes[remove] = root

def segment(filename, div):
	# Read in the image
	image = scipy.ndimage.imread(filename)[:,:,0]
	# Create the list of edges
	edges = convert(filename)
	# Get the nodes dictionary
	nodes_dict = modifiedkruskal(edges, div)
	# Count the roots and get the ten most common roots
	d = Counter(nodes_dict.values())
	segments = d.most_common(10)

	# Create numpy arrays image1, image2, and image3 such that all the pixels that are in the
	# most, second most, or third largest segments maintain their values and are set to zero
	# otherwise. The convert function might need tweaking; somehow trying to segment the original
	# image used in this lab results in the node None being the most common and nodes along the 
	# bottom row being the rest of the most common, and this doesn't seem correct.

	# Plot the images
	plt.subplot(221)
    plt.imshow(image)
    plt.subplot(222)
    plt.imshow(image1)
    plt.gray()
    plt.subplot(223)
    plt.imshow(image2)
    plt.gray()
    plt.subplot(224)
    plt.imshow(image3)
    plt.gray()
    plt.show()