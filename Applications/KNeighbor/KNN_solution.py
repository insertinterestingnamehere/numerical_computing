import numpy as np
from sklearn import neighbors


labels,points,testlabels,testpoints=np.load('PostalData.npz').items()
#Display the digit at index (0 to 49999) and print its label
# Helper functions
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

#Problem 1 uses array broadcasting to do the operation effiecently.
def weight_fun(data,weight):
    return data/wieght

#straint foward. n_neighbors=4, weights = 'distance' does the best
def Problem2():
	nbs = neighbors.KNeighborsClassifier(n_neighbors=4, weights = 'uniform', p=2)
	nbs.fit(points[1], labels[1])
	predictions=nbs.predict(testpoints[1])
	ans1=np.sum(predictions==testlabels[1])/float(len(testpoints[1]))

	nbs = neighbors.KNeighborsClassifier(n_neighbors=10, weights = 'uniform', p=2)
	nbs.fit(points[1], labels[1])
	predictions=nbs.predict(testpoints[1])
	ans2=np.sum(predictions==testlabels[1])/float(len(testpoints[1]))

	nbs = neighbors.KNeighborsClassifier(n_neighbors=4, weights = 'distance', p=2)
	nbs.fit(points[1], labels[1])
	predictions=nbs.predict(testpoints[1])
	ans3=np.sum(predictions==testlabels[1])/float(len(testpoints[1]))

	nbs = neighbors.KNeighborsClassifier(n_neighbors=10, weights = 'distance', p=2)
	nbs.fit(points[1], labels[1])
	predictions=nbs.predict(testpoints[1])
	ans4=np.sum(predictions==testlabels[1])/float(len(testpoints[1]))

	nbs = neighbors.KNeighborsClassifier(n_neighbors=1, weights = 'distance', p=2)
	nbs.fit(points[1], labels[1])
	predictions=nbs.predict(testpoints[1])
	ans5=np.sum(predictions==testlabels[1])/float(len(testpoints[1]))

	return ans1,ans2,ans3,ans4,ans5