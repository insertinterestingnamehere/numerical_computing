import scipy as sp
import matplotlib.pylab as plt
import random
from scipy import linalg as la

def kmeans(data,f,N,K,var=1,normalize=False):
	change = True
	T = data.shape[0]
	centroids = initialize(N,K,var,normalize)
	old_clusters = computeDistances(data,centroids,f)
	iters = 0
	while change:
		iters += 1
		centroids = updateCentroids(data,old_clusters,centroids,normalize)
		clusters = computeDistances(data,centroids,f)
		if sum(clusters == old_clusters) == T:
			return clusters,centroids
		print(T - sum(clusters == old_clusters))
		old_clusters = clusters

def computeDistances(data,centroids,f):
	N = centroids.shape[0]
	T = data.shape[0]
	clusterAssignments = sp.zeros(T)
	for i in xrange(T):
		dists = sp.array([f(data[i,:],centroids[j,:]) for j in xrange(N)])
		clusterAssignments[i] = sp.argmin(dists)
	return clusterAssignments

def updateCentroids(data,clusterAssignments,centroids,normalize=False):
	N = centroids.shape[0]
	for i in xrange(N):
		neighbors_i = data[clusterAssignments==i,:]
		if len(neighbors_i) != 0:
			centroids[i,:] = sp.mean(neighbors_i,0)
			if normalize:
				centroids[i,:] /= la.norm(centroids[i,:])
	return centroids

def initialize(N,K,var=1,normalize=False):
	centroids = sp.zeros((N,K))
	for i in xrange(N):
		centroids[i,:] = [random.uniform(-var,var) for j in xrange(K)]
		if normalize:
			centroids[i,:] /= la.norm(centroids[i,:])
	return centroids

def distance(x,y):
	return la.norm(x-y)

def plotFunction(data,centroids,clusters,xlim,ylim,xlabel,ylabel):
	N = centroids.shape[0]
	colors = cm.rainbow(sp.linspace(0,1,N))
	for i in xrange(N):
		plt.scatter(data[clusters==i,0],data[clusters==i,1],s=10,color=colors[i],marker='.')
	plt.scatter(centroids[:,0],centroids[:,1],s=200,c='k',marker='+')
	plt.xlim(xlim)
	plt.ylim(ylim)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.show()
	return

