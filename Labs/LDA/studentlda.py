import scipy as sp
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
			self.vocab = sp.sort(self._removeStopwords(stopwords)).tolist()
		self.documents = []
		for i in xrange(n_docs):
			self.documents.append({})
			for j in xrange(len(doclines[i])):
				if doclines[i][j] in self.vocab:
					self.documents[i][j] = self.vocab.index(doclines[i][j])

	def initialize(self):
		self.n_words = len(self.vocab)
		self.n_docs = len(self.documents)
		self.nmz = sp.zeros((self.n_docs,self.n_topics))
		self.nzw = sp.zeros((self.n_topics,self.n_words))
		self.nz = sp.zeros(self.n_topics)
		self.topics = {}
		for m in xrange(self.n_docs):
			for i in self.documents[m]:
				# Get random topic assignment
				# Increment count matrices
				# Store topic assignment

	def sample(self,filename, burnin=100, sample_rate=10, n_samples=10, stopwords=None):
		self.buildCorpus(filename,stopwords)
		self.initialize()
		self.total_nzw = sp.zeros((self.n_topics,self.n_words))
		self.total_nmz = sp.zeros((self.n_docs,self.n_topics))
		self.logprobs = sp.zeros(burnin + sample_rate*n_samples)
		for i in xrange(burnin):
			# Sweep and store log likelihood
		for i in xrange(n_samples*sample_rate):
			# Sweep and store log likelihood
			if not i%sample_rate:
				# Sweep, store log likelihood, and accumulate

	def phi(self):
		""" This function is given. """
		phi = self.total_nzw + self.beta
		self._phi = phi / sp.sum(phi, axis=1)[:,sp.newaxis]

	def theta(self):
		""" This function is given. """
		theta = self.total_nmz + self.alpha
		self._theta = theta / sp.sum(theta, axis=1)[:,sp.newaxis]

	def topterms(self,n_terms=10):
		""" This function is given. """
		vec = sp.atleast_2d(sp.arange(0,self.n_words))
		topics = []
		for k in xrange(self.n_topics):
			probs = sp.atleast_2d(self._phi[k,:])
			mat = sp.append(probs,vec,0)
			sind = sp.array([mat[:,i] for i in sp.argsort(mat[0])]).T
			topics.append([self.vocab[int(sind[1,self.n_words - 1 - i])] for i in xrange(n_terms)])
		return topics

	def toplines(self,n_lines=5):
		""" This function is given. """
		lines = sp.zeros((self.n_topics,n_lines))
		for i in xrange(self.n_topics):
			args = sp.argsort(self._theta[:,i]).tolist()
			args.reverse()
			lines[i,:] = sp.array(args)[0:n_lines] + 1
		return lines

	def _removeStopwords(self,stopwords):
		""" This function is given. """
		output = []
		for x in self.vocab:
			if x not in stopwords:
				output.append(x)
		return output

	def _conditional(self, m, w):
		# Compute conditional distribution
		return # Return conditional distribution

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
			lik += sp.sum(gammaln(self.nzw[z,:] + self.beta)) - gammaln(sp.sum(self.nzw[z,:] + self.beta))
			lik -= self.n_words * gammaln(self.beta) - gammaln(self.n_words*self.beta)

		for m in xrange(self.n_docs):
			lik += sp.sum(gammaln(self.nmz[m,:] + self.alpha)) - gammaln(sp.sum(self.nmz[m,:] + self.alpha))
			lik -= self.n_topics * gammaln(self.alpha) - gammaln(self.n_topics*self.alpha)

		return lik

