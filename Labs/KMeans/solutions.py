import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import decomposition
from scipy import linalg as la

def kmeans(data,n_clusters,init='random',max_iter=300):
    n,m = data.shape
    norms = np.zeros((n,n_clusters))
    if init=='random':
        means = np.random.randn(n_clusters,m)*data.std()
    else:
        means=init
    for i in xrange(n_clusters):
        norms[:,i] = ((data-means[i,:])**2).sum(axis=1)
    labels = np.argmin(norms,axis=1)
    for j in xrange(max_iter):
        means = np.array([data[labels==i].mean(axis=0) for i in xrange(n_clusters)])
        for i in xrange(n_clusters):
            norms[:,i] = ((data-means[i,:])**2).sum(axis=1)
        new_labels = np.argmin(norms,axis=1)
        if not np.allclose(labels,new_labels):
            labels=new_labels
        else:
            break
    inertia = 0
    for i in xrange(n_clusters):
        inertia += ((data[labels==i]-means[i,:])**2).sum()
    return means,labels,inertia
