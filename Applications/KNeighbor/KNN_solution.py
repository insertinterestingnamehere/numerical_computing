import cPickle
import scipy as sp
import matplotlib.pyplot as plt
import operator
import sys

file = 'mnist.pkl'
file = open(file, 'rb')
train_set, valid_set, test_set = cPickle.load(file)
file.close()

#Display the digit at index (0 to 49999) and print its label
def display_index(index):
#	index = 2000
	A = sp.reshape(train_set[0][index], [28,28])
	plt.imshow(A)
	print "Digit Display: " + str(train_set[1][index])


def display_test_index(index):
	A = sp.reshape(test_set[0][index], [28,28])
	plt.imshow(A)
	print "Digit Display: " + str(test_set[1][index])

#Display an average integer (0 to 9)
def display_avg_int(integer):
	#integer = 5
	#Find indices of all the 5's in the training set
	indices = [value == integer for value in train_set[1]]

	print len(indices)

	count = 0
	sum_of_images = sp.zeros(784)
	for x in range(len(indices)):
		if indices[x]:
			sum_of_images += train_set[0][x]
			count += 1
		

	sum_of_images /= count
	avg_of_images = sp.reshape(sum_of_images,[28,28])
	plt.imshow(avg_of_images,cmap='gray')
	print "There were " + str(count) + " " + str(integer) + "'s in the training set"

def calc_KNN(k, index):

	def calc_euc_dist(a,b):
	
		dist = 0
		for j in range(len(a)):
			dist += (a[j] - b[j])**2

		return sp.sqrt(dist)

	#Generate a dictionary of distances
	label_image_dict = {}
	
	for i in range(len(train_set[0])):
		dist = 0
		for j in range(len(train_set[0][i])):
			dist += (train_set[0][i][j] - test_set[0][index][j])**2
		dist = sp.sqrt(dist)
 
		label_image_dict[i] = dist
		
		sys.stdout.write('Working line: ' + str(i) + ' of ' + str(50000) + '\r')
		sys.stdout.flush()
	
	
	sorted_dict = sorted(label_image_dict.iteritems(), key=operator.itemgetter(1))

	neighbors = sp.zeros(10)
	for pair in sorted_dict[:k]:
		neighbors[train_set[1][pair[0]]] += 1

	return neighbors.argmax(), test_set[1][index]

def calc_all_KNN(k):
	#This should take a long time
	pass	
