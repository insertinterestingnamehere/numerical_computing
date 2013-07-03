import numpy as np
from scipy import linalg as la
from sklearn import cluster
from scipy.stats import norm

def logsum(arr):
    return np.log(np.sum(np.exp(arr - max(arr)))) + max(arr)

def logprop(arr):
    temp = np.exp(arr - np.max(arr))
    return temp / sum(temp)

def dmvnorm(x, mus, covars, log=False):
    if log:
        if len(mus.shape) == 1:
            return -0.5*np.dot(x - mus, la.solve(covars, x-mus)) - (len(x)/2.)*np.log(2*np.pi) - 0.5*np.log(la.det(covars))
        else:
            T = mus.shape[0]
            return np.array([-0.5*np.dot(x - mus[i,:], la.solve(covars[i,:,:], x-mus[i,:])) - (len(x)/2.)*np.log(2*np.pi) - 0.5*np.log(la.det(covars[i,:,:])) for i in xrange(T)])
    else:
        if len(mus.shape) == 1:
            return np.exp(-0.5*np.dot(x-mus, la.solve(covars, x-mus)))/((2*np.pi)**(len(x)/2.)) * np.sqrt(la.det(covars))
        else:
            T = mus.shape[0]
            return np.array([np.exp(-0.5*np.dot(x - mus[i,:], la.solve(covars[i,:,:], x - mus[i,:])))/((2*np.pi)**(len(x)/2.) * np.sqrt(la.det(covars[i,:,:]))) for i in xrange(T)])

def skl(N, model1, model2):
    data1 = model1.generate(N)
    data2 = model2.generate(N)
    return abs(np.sum(np.array([model1.dgmm(data1[i,:], log=True) - model2.dgmm(data1[i,:], log=True) for i in xrange(N)])) + np.sum(np.array([model2.dgmm(data2[i,:], log=True) - model1.dgmm(data2[i,:], log=True) for i in xrange(N)])))/(2*N)

class GMM(object):

    def __init__(self,n_components,comp=None,centers=None,covars=None):
        self.n_components = n_components
        self.comp = np.copy(comp)
        self.centers = np.copy(centers)
        self.covars = np.copy(covars)
        if centers != None:
            self.n_dim = centers.shape[1]

    def generate(self, n_sim):
        data = np.zeros((n_sim, self.n_dim))
        for i in xrange(n_sim):
            ind = np.random.multinomial(1, self.comp).argmax()
            data[i,:] = np.random.multivariate_normal(self.centers[ind,:], self.covars[ind,:,:])
        return data

    def initialize(self,data,random=False):
        self.data = data
        self.n_dim = data.shape[1]
        if random:
            mins = np.zeros(self.n_dim)
            maxes = np.zeros(self.n_dim)
            sds = np.zeros(self.n_dim)
            centers = np.zeros((self.n_components, self.n_dim))
            for i in xrange(self.n_dim):
                mins[i] = min(self.data[:, i])
                maxes[i] = max(self.data[:, i])
                sds[i] = np.std(self.data[:, i])
                centers[:, i] = np.random.uniform(mins[i], maxes[i], self.n_components)
            self.comp = np.ones(self.n_components)/float(self.n_components) + np.random.uniform(-1./self.n_components, 1./self.n_components, self.n_components)
            self.comp /= np.sum(self.comp)
            covars = np.array([np.diag(sds**2) for i in xrange(self.n_components)])
            self.centers = centers
            self.covars = covars
        else:
            clust = cluster.KMeans(self.n_components)
            clust.fit(self.data)
            self.centers = np.copy(clust.cluster_centers_)
            labels = np.copy(clust.labels_)
            self.covars = np.zeros((self.n_components, self.n_dim, self.n_dim))
            self.comp = np.zeros(self.n_components)
            for i in xrange(self.n_components):
                inds = labels == i
                temp = self.data[inds,:]
                self.covars[i,:,:] = np.dot(temp.T, temp)
                self.comp[i] = sum(inds)/float(self.data.shape[0])

    def dgmm(self,x,pieces=False,log=False):
        if pieces:
            if log:
                return np.log(self.comp) + dmvnorm(x, self.centers, self.covars, log=True)
            else:
                return self.comp*dmvnorm(x, self.centers, self.covars)
        else:
            if log:
                return logsum(np.log(self.comp) + dmvnorm(x, self.centers, self.covars, log=True))
            else:
                return np.dot(self.comp, dmvnorm(x, self.centers, self.covars))

    def train(self,data,tol=.1,random=False):
        self.initialize(data, random=random)
        logprobs = []
        i = 0
        diff = 100
        logprob = self._gamma()
        while diff > tol:
            self._reestimate_means()
            self._reestimate_covars()
            self._reestimate_comp()
            new_logprob = self._gamma()
            if i > -1:
                diff = new_logprob - logprob
            i += 1
            logprob = new_logprob
            print "Iteration: {}".format(i) + "\tLog-Prob: {}".format(logprob)

    def _gamma(self):
        T = self.data.shape[0]
        gam = np.zeros((T, self.n_components))
        logprob = 0
        for i in xrange(T):
            temp = self.dgmm(self.data[i,:], pieces=True, log=True)
            gam[i,:] = logprop(temp)
            logprob += logsum(temp)
        self.gam = gam
        return logprob

    def _reestimate_means(self):
        self.centers = np.dot(self.gam.T, self.data)/np.sum(self.gam, 0)[:, np.newaxis]

    def _reestimate_covars(self):
        for i in xrange(self.n_components):
            temp = self.gam[:, i][:, np.newaxis] * (self.data - self.centers[i,:])
            self.covars[i,:,:] = np.dot(temp.T, self.data - self.centers[i,:]) / np.sum(self.gam[:, i])

    def _reestimate_comp(self):
        self.comp = np.sum(self.gam, 0)
        self.comp /= np.sum(self.comp)
