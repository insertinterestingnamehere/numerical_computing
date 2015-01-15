import scipy as sp
import scipy.linalg as la
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D
import warnings
from sklearn.decomposition import PCA

cols = ['b', 'g', 'r']

class KNN(object):

    def __init__(self, training_data, training_labels):
        if training_data.shape[0] != len(training_labels):
            raise ValueError('There must be the same number of training points as training labels.')
        self.training_data = training_data
        self.training_labels = training_labels
        self.classes = list(set(training_labels))
        self.n_classes = len(self.classes)
        self.alias_classes = sp.arange(self.n_classes)
        self.aliases = sp.zeros(len(training_labels))
        for i in xrange(self.n_classes):
            bools = training_labels == self.classes[i]
            self.aliases[bools] = sp.ones(sum(bools))*self.alias_classes[i]

    def classify(self, test_data, n_neighbors=5):
        if test_data.shape[1] != self.training_data.shape[1]:
            raise ValueError('Training data and test data do not have the same dimensions.')
        n_train = self.training_data.shape[0]
        n_test = test_data.shape[0]
        dists = sp.zeros((n_test, n_train))
        labels = []
        for i in xrange(n_test):
            for j in xrange(n_train):
                dists[i, j] = la.norm(self.training_data[j,:] - test_data[i,:])
            inds = sp.argsort(dists[i,:])[0:n_neighbors]
            votes = sp.array([sum(self.training_labels[inds] == self.classes[k]) for k in xrange(self.n_classes)])
            labels.append(self.classes[sp.copy(votes).argmax()])
        return labels

    def plot(self, test_data, labels):
        if test_data.shape[0] != len(labels):
            raise ValueError('There must be the same number of test points as test labels.')
        if test_data.shape[1] != self.training_data.shape[1]:
            raise ValueError('Training data and test data do not have the same dimensions.')
        if test_data.shape[1] <= 1:
            raise ValueError('Don\'t be silly.')
        if test_data.shape[1] > 3:
            print "Too many dimensions - Reducing via PCA."
            n_comp = int(raw_input('How many components (2 or 3)? \n'))
            while n_comp != 2 and n_comp != 3:
                n_comp = int(raw_input('How many components (2 or 3)? \n'))
            pca = PCA(n_components=n_comp)
            dat = sp.append(sp.copy(self.training_data), sp.copy(test_data), axis=0)
            pca.fit(dat)
            test_data = pca.transform(test_data)
            training_data = pca.transform(self.training_data)
        else:
            training_data = self.training_data
        if test_data.shape[1] == 2:
            for i in xrange(self.n_classes):
                inds = self.training_labels == self.classes[i]
                plt.plot(training_data[inds, 0], training_data[inds, 1], cols[i]+'.')
            for i in xrange(self.n_classes):
                inds = labels == self.classes[i]
                plt.plot(test_data[inds, 0], test_data[inds, 1], cols[i]+'*')
        else:
            fig = pylab.figure()
            ax = Axes3D(fig)
            for i in xrange(self.n_classes):
                inds = self.training_labels == self.classes[i]
                ax.scatter(training_data[inds, 0], training_data[inds, 1], training_data[inds, 2], c=cols[i], marker='.')
            for i in xrange(self.n_classes):
                inds = labels == self.classes[i]
                ax.scatter(test_data[inds, 0], test_data[inds, 1], test_data[inds, 2], c=cols[i], marker='*')
        plt.show()
