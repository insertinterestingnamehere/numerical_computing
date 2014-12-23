import numpy as np
from numpy.random import rand
from scipy.spatial import KDTree
import timeit
from sklearn import neighbors


# solution to problem 1
def nearestNNaive(points, x):
    l = len(points)
    r = sum((x - points[0]) ** 2)
    point = 0
    for i in xrange(l):
        d = sum((x - points[i]) ** 2)
        if d < r:
            r = d
            point = i
    return r ** (.5), point


# builds a kdtree
class Node(object):
    pass


def kdtree(points, depth=0):

    if len(points) == 0:
        return None

    k = len(points[0])
    axis = depth % k

    points = points.take(points[:,axis].argsort(), axis=0)
    median = len(points) / 2

    # Create node and construct subtrees
    node = Node()
    node.location = points[median]
    node.left_child = kdtree(points[:median], depth + 1)
    node.right_child = kdtree(points[median + 1:], depth + 1)
    return node


# Helper function to KDstart. searches the kd-tree using recursion,
# Algortihm can problaly simplified.
def KDsearch(node, point, best, bpoint, depth=0):
    if node is None:
        return best, bpoint

    k = len(node.location)
    axis = depth % k
    d = sum((point - node.location) ** 2)
    if d < best:
        best = d
        bpoint = node.location[:]
    if point[axis] < node.location[axis]:
        best, bpoint = KDsearch(
            node.left_child, point, best, bpoint, depth + 1)
        if point[axis] + best >= node.location[axis]:
            best, bpoint = KDsearch(
                node.right_child, point, best, bpoint, depth + 1)
    else:
        best, bpoint = KDsearch(
            node.right_child, point, best, bpoint, depth + 1)
        if point[axis] - best <= node.location[axis]:
            best, bpoint = KDsearch(
                node.left_child, point, best, bpoint, depth + 1)

    return best, bpoint


# Starts the search of the KD-tree.
def KDstart(tree, point):
    best, bpoint = KDsearch(
        tree, point, sum((point - tree.location) ** 2), tree.location)
    return best ** (.5), bpoint


# timer function used to find the times for problems 3-5
def timeFun(f, *args, **kargs):
    pfunc = lambda: f(*args, **kargs)
    theTime = timeit.Timer(pfunc)
    return min(theTime.repeat(1, 1))


labels, points, testlabels, testpoints = np.load('PostalData.npz').items()


# Display the digit at index (0 to 49999) and print its label
# Helper functions
def display_index(index):
    A = sp.reshape(train_set[0][index], [28, 28])
    plt.imshow(A)
    print "Digit Display: " + str(train_set[1][index])


def display_test_index(index):
    A = sp.reshape(test_set[0][index], [28, 28])
    plt.imshow(A)
    print "Digit Display: " + str(test_set[1][index])


# Display an average integer (0 to 9)
def display_avg_int(integer):
    # integer = 5
    # Find indices of all the 5's in the training set
    indices = [value == integer for value in train_set[1]]

    print len(indices)

    count = 0
    sum_of_images = sp.zeros(784)
    for x in range(len(indices)):
        if indices[x]:
            sum_of_images += train_set[0][x]
            count += 1

    sum_of_images /= count
    avg_of_images = sp.reshape(sum_of_images, [28, 28])
    plt.imshow(avg_of_images, cmap='gray')
    print "There were " + str(count) + " " + str(integer) + "'s in the training set"


# Problem 7 uses array broadcasting to do the operation effiecently.
def weight_fun(data, weight):
    return data / wieght


# Project
# straint foward. n_neighbors=4, weights = 'distance' does the best
def Problem7():
    nbs = neighbors.KNeighborsClassifier(n_neighbors=4, weights='uniform', p=2)
    nbs.fit(points[1], labels[1])
    predictions = nbs.predict(testpoints[1])
    ans1 = np.sum(predictions == testlabels[1]) / float(len(testpoints[1]))

    nbs = neighbors.KNeighborsClassifier(
        n_neighbors=10, weights='uniform', p=2)
    nbs.fit(points[1], labels[1])
    predictions = nbs.predict(testpoints[1])
    ans2 = np.sum(predictions == testlabels[1]) / float(len(testpoints[1]))

    nbs = neighbors.KNeighborsClassifier(
        n_neighbors=4, weights='distance', p=2)
    nbs.fit(points[1], labels[1])
    predictions = nbs.predict(testpoints[1])
    ans3 = np.sum(predictions == testlabels[1]) / float(len(testpoints[1]))

    nbs = neighbors.KNeighborsClassifier(
        n_neighbors=10, weights='distance', p=2)
    nbs.fit(points[1], labels[1])
    predictions = nbs.predict(testpoints[1])
    ans4 = np.sum(predictions == testlabels[1]) / float(len(testpoints[1]))

    nbs = neighbors.KNeighborsClassifier(
        n_neighbors=1, weights='distance', p=2)
    nbs.fit(points[1], labels[1])
    predictions = nbs.predict(testpoints[1])
    ans5 = np.sum(predictions == testlabels[1]) / float(len(testpoints[1]))

    return ans1, ans2, ans3, ans4, ans5
