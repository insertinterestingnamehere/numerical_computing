#This is a kdtree written by A Zaitzeff to be used by the students for problem 2 of NNS
class Node: 
    pass
 
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