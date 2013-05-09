import scipy as sp
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA

digits = datasets.load_digits()
data = digits.data
target = digits.target
N = len(target)

inds = random.sample(sp.arange(0,N),N)
n_train = int(sp.floor(0.8*N))
trainingdata = data[inds[0:n_train],:]
trainingtarget = target[inds[0:n_train]]
testdata = data[inds[n_train:]]
testtarget = target[inds[n_train:]]

KNN = KNeighborsClassifier(n_neighbors=5)
KNN.fit(trainingdata,trainingtarget)
predictions = KNN.predict(testdata)
n_misclassed = sum(predictions!=testtarget)
misclassrate = n_misclassed/float(N-n_train)
print misclassrate

param_grid = {'C': [1e3,5e3,1e4,5e4,1e5],'gamma': [0.0001,0.0005,0.001,0.005,0.01,0.1],}
clf = GridSearchCV(SVC(kernel='rbf',class_weight='auto'), param_grid)
clf = clf.fit(trainingdata,trainingtarget)

predictions = clf.predict(testdata)
print sum(predictions!=testtarget)/float(len(testtarget))

people = fetch_lfw_people(min_faces_per_person=70,resize=0.4)
n_samples,h,w = lfw_people.images.shape

data = lfw_people.data
n_features = data.shape[1]

target = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

N = len(target)
inds = random.sample(sp.arange(0,N),N)
n_train = int(sp.floor(0.8*N))
trainingdata = data[inds[0:n_train],:]
trainingtarget = target[inds[0:n_train]]
testdata = data[inds[n_train:]]
testtarget = target[inds[n_train:]]

n_components = 150
pca = PCA(n_components=n_components,whiten=True).fit(trainingdata)
trainingdata_pca = pca.transform(trainingdata)
testdata_pca = pca.transform(testdata)

clf = GridSearchCV(SVC(kernel='rbf',class_weight='auto'), param_grid)
clf = clf.fit(trainingdata_pca,trainingtarget)

predictions = clf.predict(testdata_pca)
print sum(predictions!=testtarget)/float(len(testtarget))

KNN = KNeighborsClassifier(n_neighbors=5)
KNN.fit(trainingdata_pca,trainingtarget)
predictions = KNN.predict(testdata_pca)
print sum(predictions!=testtarget)/float(len(testtarget))
