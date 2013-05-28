import scipy as sp
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA

def gen_digits_sets():
        
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
    
    return trainingdata, testdata, trainingtarget, testtarget

def misclass(trainingdata, testdata, trainingtarget, testtarget):
        
    KNN = KNeighborsClassifier(n_neighbors=5)
    KNN.fit(trainingdata,trainingtarget)
    predictions = KNN.predict(testdata)
    n_misclassed = sum(predictions!=testtarget)
    misclassrate = n_misclassed/float(len(testtarget)+len(trainingtarget)-len(trainingdata))
    return misclassrate

def svm_knn(trainingdata, testdata, trainingtarget, testtarget):
    param_grid = {'C': [1e3,5e3,1e4,5e4,1e5], 
                  'gamma': [0.0001,0.0005,0.001,0.005,0.01,0.1]}
    clf = GridSearchCV(SVC(kernel='rbf',class_weight='auto'), param_grid)
    clf = clf.fit(trainingdata,trainingtarget)

    predictions = clf.predict(testdata)
    return sum(predictions!=testtarget)/float(len(testtarget))

def gen_face_sets():
    people = fetch_lfw_people(min_faces_per_person=70,resize=0.4)
    n_samples,h,w = people.images.shape

    data = people.data
    n_features = data.shape[1]

    target = people.target
    target_names = people.target_names
    n_classes = target_names.shape[0]


    N = len(target)
    inds = random.sample(sp.arange(0,N),N)
    n_train = int(sp.floor(0.8*N))
    trainingdata = data[inds[0:n_train],:]
    trainingtarget = target[inds[0:n_train]]
    testdata = data[inds[n_train:]]
    testtarget = target[inds[n_train:]]
    return trainingdata, testdata, trainingtarget, testtarget

def pca_trans(trainingdata, testdata):
    n_components = 150
    pca = PCA(n_components=n_components, whiten=True).fit(trainingdata)
    trainingdata_pca = pca.transform(trainingdata)
    return pca.transform(testdata)
    
def prob5(trainingdata, testdata, trainingtarget, testtarget):
    trainingdata_pca = pca_trans(trainingdata, testdata)
    param_grid = {'C': [1e3,5e3,1e4,5e4,1e5],
                  'gamma': [0.0001,0.0005,0.001,0.005,0.01,0.1]}
    clf = GridSearchCV(SVC(kernel='rbf',class_weight='auto'), param_grid)
    clf = clf.fit(trainingdata_pca,trainingtarget)

    predictions = clf.predict(testdata_pca)
    return sum(predictions!=testtarget)/float(len(testtarget))

def prob6(trainingdata, testdata, trainingtarget, testtarget):
    trainingdata_pca = pca_trans(trainingdata, testdata)
    KNN = KNeighborsClassifier(n_neighbors=5)
    KNN.fit(trainingdata_pca,trainingtarget)
    predictions = KNN.predict(testdata_pca)
    return sum(predictions!=testtarget)/float(len(testtarget))
