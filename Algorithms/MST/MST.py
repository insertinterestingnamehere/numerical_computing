import numpy as np
from operator import itemgetter

def kruskal(edges):
    # Initialize an empty list of edges for the MST.
    tree = []
    # Make a dictionary that points each node
    # toward its root (not always directly to it).
    # Start with each node pointing to itself.
    nodes = {node:node for node in ({edge[0] for edge in edges} | {edge[1] for edge in edges})}
    # Initialize the number of nodes that still need
    # to be processed to the number of nodes minus 1.
    remaining = len(nodes) - 1
    # Define a helper function that, given a node, traces
    # through the dictionary to find the root of its tree.
    def track(node):
        # Initialize a temporary variable to be the
        # node for which we are finding the root.
        temp = node
        # While the temporary node does not
        # point to itself in the dictionary:
        while nodes[temp] is not temp:
            # Update the temporary node to be the node
            # it currently points to in the dictionary.
            temp = nodes[temp]
        # Update the temporary node to be the node it currently points to in the dictionary.
        return temp
    # Iterate over the edges by ascending weight.
    for n1, n2, weight in sorted(edges, key=itemgetter(2)):
        # Trace through the dictionary to find the root node
        # of each of the nodes in the edge you are processing.
        root = track(n1)
        remove = track(n2)
        # If the roots are not the same (i.e.
        # if adding the edge doesn't form a cycle):
        if root is not remove:
            # Add the edge to the tree.
            tree.append((n1, n2, weight))
            # Lower the number of edges remaining by one.
            remaining -= 1
            # If the number of edges remaining is 0,
            # return the tree (which also breaks the loop).
            if remaining == 0:
                return tree
            # Update the root of the root of the second node in
            # the edge to be the root of the first node in the edge.
            # This lets us know that the two subtrees are connected later on.
            nodes[remove] = root

def prim(edges):
    # Initialize a dictionary to track which nodes have been processed.
    nodes = {node:True for node in {edge[0] for edge in edges} | {edge[1] for edge in edges}}
    # Initialize an empty dictionary of lists to track the edges containing each node.
    edgelist = {node:[] for node in nodes}
    # Fill the edge list.
    for n1, n2, weight in edges:
        # Add each edge to the list corresponding to both of its nodes.
        edgelist[n1].append((n2,weight))
        edgelist[n2].append((n1,weight))
    # Get the first edge to add (It can be the shortest edge
    # from any given node, the shortest edge is a good pick).
    add = min(edges, key=itemgetter(2))
    # Mark the nodes in the first edge as processed.
    nodes[add[0]] = False
    nodes[add[1]] = False
    # Initialize the tree to be the list containing the first edge.
    tree = [add]
    # Initialize an empty dictionary that will be used
    # to contain the next edges that can be processed.
    # note: we will store full edges in this dict
    # always put the node just processed first
    vals = {}
    # Define a helper function to insert an edge into
    # the dictionary  if that insertion is needed.
    def insert(outside, node, weight):
        if nodes[node]:
            # Get the value of the node that is reached by the edge.
            val = vals.get(node)
            # If that node isn't in the dictionary, set its
            # value to be the edge passed to the functions.
            if val is None:
                vals[node] = (outside, node, weight)
            # If it is in the dictionary already, set its value
            # to be the shorter of the edge being processed
            # and the edge already in the dictionary.
            elif weight < vals[node][2]:
                vals[node] = (outside, node, weight)
    # Use the helper function to insert the edges
    # reached by the first two processed nodes
    # into the dictionary of edges to be processed.
    for node, weight in edgelist[add[0]]:
        insert(add[0], node, weight)
    for node, weight in edgelist[add[1]]:
        insert(add[0], node, weight)
    # Until the tree contains enough edges to span all the nodes:
    while len(tree) < len(nodes) - 1:
        # Find the shortest edge in the dictionary of edges to be processed.
        inside, outside, weight = min(vals.itervalues(), key=itemgetter(2))
        # Remove the shortest edge from the dictionary.
        del vals[outside]
        # Add it to the tree.
        tree.append((inside, outside, weight))
        # Mark the node reached by the new edge as processed.
        nodes[outside] = False
        # Use the helper function to insert the edges
        # reached by the newly processed node into
        # the dictionary of edges to be processed.
        for node, weight in edgelist[outside]:
            insert(outside, node, weight)
    # Return the completed tree.
    return tree

# Here is a faster version of Prim's algorithm in case we need it at some point.

def primfast(edges):
    nodelist = list({edge[0] for edge in edges} | {edge[1] for edge in edges})
    size = len(nodelist)
    end = size - 1
    reverse = {nodelist[i]:i for i in xrange(size)}
    nodes = [True] * size
    edgelist = [[] for i in xrange(size)]
    add = edges[0]
    addweight = add[2]
    for n1, n2, weight in edges:
        node1, node2 = reverse[n1], reverse[n2]
        if node1 != node2:
            edgelist[node1].append((node2,weight))
            edgelist[node2].append((node1,weight))
        if weight < addweight:
            add = (n1, n2, weight)
            addweight = weight
    tree = [add]
    vals = {}
    node0 = reverse[add[0]]
    node1 = reverse[add[1]]
    nodes[node0] = False
    nodes[node1] = False
    for node, weight in edgelist[node0]:
        if node is not node1:
            vals[node] = (node0, node, weight)
    for node, weight in edgelist[node1]:
        if node is not node0:
            val = vals.get(node)
            if val is None:
                vals[node] = (node1, node, weight)
            elif weight < val[2]:
                vals[node] = (node1, node, weight)                
    while len(tree) < end:
        inside, outside, weight = min(vals.itervalues(), key=itemgetter(2))
        del vals[outside]
        tree.append((nodelist[inside], nodelist[outside], weight))
        nodes[outside] = False
        for node, weight in edgelist[outside]:
            if nodes[node]:
                val = vals.get(node)
                if val is None:
                    vals[node] = (outside, node, weight)
                elif weight < vals[node][2]:
                    vals[node] = (outside, node, weight)                    
    return tree

def formChanger(oldData):
    newData=[]
    for i in oldData:
        newData.append((i[0],i[1],int(i[2])))
    return newData

# example usage
# q=np.load("MSTdata.npy")
# edges=formChanger(q)
# kruskal(edges)
# prim(edges)
