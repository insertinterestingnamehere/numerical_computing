import numpy as np
from scipy.special import gammaln
import string

def loadStopwords(filename):
	""" This function is given. """
	infile = open(filename,'r')
	stopwords = infile.readlines()
	for i in xrange(len(stopwords)):
		stopwords[i] = stopwords[i].rstrip()
	stopwords.append('')
	return stopwords

class LDACGS(object):

	def __init__(self, n_topics, alpha = 0.1, beta = 0.1):
		""" This function is given. """
		self.n_topics = n_topics
		self.alpha = alpha
		self.beta = beta

	def buildCorpus(self,filename,stopwords=None):
		""" This function is given. """
		infile = open(filename,'r')
		doclines = [line.rstrip().translate(string.maketrans("",""),string.punctuation).lower().split(' ') for line in infile]
		n_docs = len(doclines)
		self.vocab = []
		for i in xrange(n_docs):
			self.vocab += doclines[i]
		self.vocab = list(set(self.vocab))
		if stopwords != None:
			self.vocab = np.sort(self._removeStopwords(stopwords)).tolist()
		self.documents = []
		for i in xrange(n_docs):
			self.documents.append({})
			for j in xrange(len(doclines[i])):
				if doclines[i][j] in self.vocab:
					self.documents[i][j] = self.vocab.index(doclines[i][j])

	def initialize(self):
		self.n_words = len(self.vocab)
		self.n_docs = len(self.documents)

        # initialize the three count matrices
        # the (i,j) entry of self.nmz is # of words in document i assigned to topic j
		self.nmz = np.zeros((self.n_docs,self.n_topics))
        # the (i,j) entry of self.nzw is # of times term j is assigned to topic i
		self.nzw = np.zeros((self.n_topics,self.n_words))
        # the (i)-th entry is the number of times topic i is assigned in the corpus
		self.nz = np.zeros(self.n_topics)

        # initialize the topic assignment dictionary
		self.topics = {} # key-value pairs of form (m,i):z
		for m in xrange(self.n_docs):
			for i in self.documents[m]:
				# Get random topic assignment, i.e. z = ...
				# Increment count matrices
				# Store topic assignment, i.e. self.topics[(m,i)]=z

	def sample(self,filename, burnin=100, sample_rate=10, n_samples=10, stopwords=None):
		self.buildCorpus(filename,stopwords)
		self.initialize()
		self.total_nzw = np.zeros((self.n_topics,self.n_words))
		self.total_nmz = np.zeros((self.n_docs,self.n_topics))
		self.logprobs = np.zeros(burnin + sample_rate*n_samples)
		for i in xrange(burnin):
			# Sweep and store log likelihood
		for i in xrange(n_samples*sample_rate):
			# Sweep and store log likelihood
			if not i%sample_rate:
				# accumulate counts

	def phi(self):
		""" This function is given. """
		phi = self.total_nzw + self.beta
		self._phi = phi / np.sum(phi, axis=1)[:,np.newaxis]

	def theta(self):
		""" This function is given. """
		theta = self.total_nmz + self.alpha
		self._theta = theta / np.sum(theta, axis=1)[:,np.newaxis]

	def topterms(self,n_terms=10):
		""" This function is given. """
		vec = np.atleast_2d(np.arange(0,self.n_words))
		topics = []
		for k in xrange(self.n_topics):
			probs = np.atleast_2d(self._phi[k,:])
			mat = np.append(probs,vec,0)
			sind = np.array([mat[:,i] for i in np.argsort(mat[0])]).T
			topics.append([self.vocab[int(sind[1,self.n_words - 1 - i])] for i in xrange(n_terms)])
		return topics

	def toplines(self,n_lines=5):
		""" This function is given. """
		lines = np.zeros((self.n_topics,n_lines))
		for i in xrange(self.n_topics):
			args = np.argsort(self._theta[:,i]).tolist()
			args.reverse()
			lines[i,:] = np.array(args)[0:n_lines] + 1
		return lines

	def _removeStopwords(self,stopwords):
		""" This function is given. """
		output = []
		for x in self.vocab:
			if x not in stopwords:
				output.append(x)
		return output

	def _conditional(self, m, w):
        """ 
        This function is given. Compute the conditional distribution of 
        the topic corresponding to document m and word index w.
        Returns a distribution vector of length self.n_topics.
        """
		dist = (self.nmz[m,:] + self.alpha) * (self.nzw[:,w] + self.beta) / (self.nz + self.beta*self.n_words)
        return dist/sum(dist)

	def _sweep(self):
		for m in xrange(self.n_docs):
			for i in self.documents[m]:
				# Retrieve vocab index for i^th word in document m
				# Retrieve topic assignment for i^th word in document m
				# Decrement count matrices
				# Get conditional distribution
				# Sample new topic assignment
				# Increment count matrices
				# Store new topic assignment

	def _loglikelihood(self):
		""" This function is given. """
		lik = 0

		for z in xrange(self.n_topics):
			lik += np.sum(gammaln(self.nzw[z,:] + self.beta)) - gammaln(np.sum(self.nzw[z,:] + self.beta))
			lik -= self.n_words * gammaln(self.beta) - gammaln(self.n_words*self.beta)

		for m in xrange(self.n_docs):
			lik += np.sum(gammaln(self.nmz[m,:] + self.alpha)) - gammaln(np.sum(self.nmz[m,:] + self.alpha))
			lik -= self.n_topics * gammaln(self.alpha) - gammaln(self.n_topics*self.alpha)

		return lik

