import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import scipy.stats as st
from matplotlib import pyplot as plt

def Gaussian_NB(features, means, variances, prior):
    """
    Classify the feature vectors using a Gaussian Naive Bayes model.

    Parameters
    ----------
    features : ndarray of shape (m,n)
        Each row is a feature vector, each column a feature
    means : ndarray of shape (n,k)
        Mean of each feature (corresponding to columns of features)
    variances : ndarray of shape (n,k)
        Variance of each feature
    prior : ndarray of shape (k,)
        Prior probability of each of k classes

    Returns
    -------
    labels : ndarray of shape (m,)
        The ith entry gives the class of feature vector i as a number in {0,1,...,k-1}.
    """
    m, n = features.shape
    k = len(prior)
    stds = np.sqrt(variances)
    logprobs = np.zeros((m,k))
    logprobs += np.log(prior)
    for i in xrange(m):
        for j in xrange(k):
            for l in xrange(n):
                logprobs[i,j] += st.norm.logpdf(features[i,l], loc=means[l,j], scale=stds[l,j])
    return logprobs.argmax(axis=1)

# seed classification problem    
def seeds():
    data = pd.read_csv("seeds_dataset.txt",  delim_whitespace=True, names=
            "Area, Perimeter, Compactness, Length, Width, Asymmetry Coefficient, Groove Length, Class".split(", "))
    test = data.loc[np.random.choice(data.index,40, replace=False)]
    train = data.loc[[i for i in set.difference(set(data.index), set(test.index))]]
    means = np.zeros((7,3))
    variances = np.zeros((7,3))
    for i in xrange(3):
        means[:,i] = np.array(train[train["Class"]==i+1].mean())[:-1]
        variances[:,i] = np.array(train[train["Class"]==i+1].var())[:-1]
    prior = np.ones(3)/3.
    labels = Gaussian_NB(np.array(test)[:,:-1], means, variances, prior) + 1
    truth = np.array(test)[:,-1]
    print "Accuracy:", (labels==truth).sum()/float(len(labels))
    nb = GaussianNB()
    nb.fit(np.array(train)[:,:-1], np.array(train)[:,-1])
    labels2 = nb.predict(np.array(test)[:,:-1])
    print np.allclose(labels, labels2)

class naiveBayes(object):
    """
    This class performs naive bayes classification for word-count document features.
    """
    def __init__(self):
        """
        Initialize a naive Bayes classifier.
        """
        self.n_samples, self.n_features, self.class_probs, self.n_classes, self.word_class_probs = [None]*5

    def fit(self,X,Y):
        """
        Fit the parameters according to the labeled training data (X,Y).

        Parameters
        ----------
        X : numpy arrary of shape [n_samples, n_features].
            Each row is essentially the "word-count" vector for one of the "documents".
        Y : numpy array of shape [n_samples].
            Gives the class label for each instance of training data. Assume class labels
            are {0,1,...,k-1} where k is the number of classes.
        """
        # get number of samples, number of features (i.e. size of "vocabulary")
        self.n_samples, self.n_features = X.shape

        # get MLE estimates of class probabilities
        self.class_probs = np.array([(Y==i).sum() for i in set(Y)])/float(self.n_samples)
        self.n_classes = len(self.class_probs)

        # get (smoothed) MLE estimates of word-class vocabularies
        # probability of word i given class j is the (i,j)-th entry of the matrix
        self.word_class_probs = np.empty([self.n_classes,self.n_features])
        for c in xrange(self.n_classes):
            self.word_class_probs[c,:] = (X[Y==c].sum(axis=0)+1).T
            self.word_class_probs[c,:] /= self.word_class_probs[c,:].sum()

    def predict(self, X):
        """
        Predict the class labels of a set of test data.

        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]

        Returns
        -------
        Y : numpy array of shape [n_samples].
            Gives the classification of each row in X
        """
        return np.argmax(np.log(self.class_probs) + X.dot(np.log(self.word_class_probs).T), axis=1)
        
# spam filtering
def spam():
    dat = np.loadtxt("SpamFeatures.txt")
    labs = np.loadtxt("SpamLabels.txt")
    nb = naiveBayes()
    test_rows = np.random.choice(np.arange(len(labs)),500, replace=False)
    train_rows = np.array([i for i in xrange(len(labs)) if i not in test_rows])
    nb.fit(dat[train_rows], labs[train_rows])
    new_labs = nb.predict(dat[test_rows])
    print (new_labs==labs[test_rows]).sum()/float(len(new_labs))
    mnb = MultinomialNB()
    mnb.fit(dat[train_rows], labs[train_rows])
    new_labs2 = mnb.predict(dat[test_rows])
    print (new_labs2 == labs[test_rows]).sum()/float(len(new_labs))
