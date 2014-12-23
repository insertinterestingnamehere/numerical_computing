import scipy as sp
import numpy as np
from scipy import linalg as la
from sklearn import cluster
from scipy.stats import norm

def logsum(arr):
	return sp.log(sp.sum(sp.exp(arr - max(arr)))) + max(arr)

def logprop(arr):
	temp = sp.exp(arr - np.max(arr))
	return temp / sum(temp)

def dmvnorm(x,mus,covars,log=False):
	if log:
		if len(mus.shape) == 1:
			return -0.5*sp.dot(x - mus,la.solve(covars,x-mus)) - (len(x)/2.)*sp.log(2*sp.pi) - 0.5*sp.log(la.det(covars))
		else:
			T = mus.shape[0]
			return sp.array([-0.5*sp.dot(x - mus[i,:],la.solve(covars[i,:,:],x-mus[i,:])) - (len(x)/2.)*sp.log(2*sp.pi) - 0.5*sp.log(la.det(covars[i,:,:])) for i in xrange(T)])
	else:
		if len(mus.shape) == 1:
			return sp.exp(-0.5*sp.dot(x-mus,la.solve(covars,x-mus)))/((2*sp.pi)**(len(x)/2.)) * sp.sqrt(la.det(covars))
		else:
			T = mus.shape[0]
			return sp.array([sp.exp(-0.5*sp.dot(x - mus[i,:],la.solve(covars[i,:,:],x - mus[i,:])))/((2*sp.pi)**(len(x)/2.) * sp.sqrt(la.det(covars[i,:,:]))) for i in xrange(T)])

def skl(N,model1,model2):
	data1 = model1.generate(N)
	data2 = model2.generate(N)
	return abs(sp.sum(sp.array([model1.dgmm(data1[i,:],log=True) - model2.dgmm(data1[i,:],log=True) for i in xrange(N)])) + sp.sum(sp.array([model2.dgmm(data2[i,:],log=True) - model1.dgmm(data2[i,:],log=True) for i in xrange(N)])))/(2*N)

class GMM(object):

	def __init__(self,n_components,comp=None,centers=None,covars=None):
		self.n_components = n_components
		self.comp = sp.copy(comp)
		self.centers = sp.copy(centers)
		self.covars = sp.copy(covars)
		if centers != None:
			self.n_dim = centers.shape[1]

	def generate(self,n_sim):
                data = sp.zeros((n_sim,self.n_dim))
                for i in xrange(n_sim):
                        ind = sp.random.multinomial(1,self.comp).argmax()
                        data[i,:] = sp.random.multivariate_normal(self.centers[ind,:],self.covars[ind,:,:])
                return data

	def initialize(self,data,random=False):
		self.data = data
		self.n_dim = data.shape[1]
		if random:
			mins = sp.zeros(self.n_dim)
			maxes = sp.zeros(self.n_dim)
			sds = sp.zeros(self.n_dim)
			centers = sp.zeros((self.n_components,self.n_dim))
			for i in xrange(self.n_dim):
				mins[i] = min(self.data[:,i])
				maxes[i] = max(self.data[:,i])
				sds[i] = sp.std(self.data[:,i])
				centers[:,i] = sp.random.uniform(mins[i],maxes[i],self.n_components)
			self.comp = sp.ones(self.n_components)/float(self.n_components) + sp.random.uniform(-1./self.n_components,1./self.n_components,self.n_components)
			self.comp /= sp.sum(self.comp)
			covars = sp.array([sp.diag(sds**2) for i in xrange(self.n_components)])
			self.centers = centers
			self.covars = covars
		else:
			clust = cluster.KMeans(self.n_components)
			clust.fit(self.data)
			self.centers = sp.copy(clust.cluster_centers_)
			labels = sp.copy(clust.labels_)
			self.covars = sp.zeros((self.n_components,self.n_dim,self.n_dim))
			self.comp = sp.zeros(self.n_components)
			for i in xrange(self.n_components):
				inds = labels == i
				temp = self.data[inds,:]
				self.covars[i,:,:] = sp.dot(temp.T,temp)
				self.comp[i] = sum(inds)/float(self.data.shape[0])

	def dgmm(self,x,pieces=False,log=False):
		if pieces:
			if log:
				return sp.log(self.comp) + dmvnorm(x,self.centers,self.covars,log=True)
			else:
				return self.comp*dmvnorm(x,self.centers,self.covars)
		else:
			if log:
				return logsum(sp.log(self.comp) + dmvnorm(x,self.centers,self.covars,log=True))
			else:
				return sp.dot(self.comp,dmvnorm(x,self.centers,self.covars))

	def train(self,data,tol=.1,random=False):
		self.initialize(data,random=random)
		logprobs = []
		i = 0
		diff = 100
		logprob = self._gamma()
		while diff > tol:
			self._reestimate_means()
			self._reestimate_covars()
			self._reestimate_comp()
			new_logprob = self._gamma()
			if i > 10:
				diff = new_logprob - logprob
			i += 1
			logprob = new_logprob
			print "Iteration: {}".format(i) + "\tLog-Prob: {}".format(logprob)

	def _gamma(self):
		T = self.data.shape[0]
		gam = sp.zeros((T,self.n_components))
		logprob = 0
		for i in xrange(T):
			temp = self.dgmm(self.data[i,:],pieces=True,log=True)
			gam[i,:] = logprop(temp)
			logprob += logsum(temp)
		self.gam = gam
		return logprob

	def _reestimate_means(self):
		self.centers = sp.dot(self.gam.T,self.data)/sp.sum(self.gam,0)[:,sp.newaxis]

	def _reestimate_covars(self):
		for i in xrange(self.n_components):
			temp = self.gam[:,i][:,sp.newaxis] * (self.data - self.centers[i,:])
			self.covars[i,:,:] = sp.dot(temp.T,self.data - self.centers[i,:]) / sp.sum(self.gam[:,i])

	def _reestimate_comp(self):
		self.comp = sp.sum(self.gam,0)
		self.comp /= sp.sum(self.comp)
