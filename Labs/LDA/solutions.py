import numpy as np
import scipy.stats as st
from math import sqrt
import scipy as sp
from scipy.special import gammaln
import string

def gibbs(y, mu0, sigma02, alpha, beta, n_samples):
    """
    Assuming a likelihood and priors
        y_i    ~ N(mu, sigma2),
        mu     ~ N(mu0, sigma02),
        sigma2 ~ IG(alpha, beta),
    sample from the posterior distribution
        P(mu, sigma2 | y, mu0, sigma02, alpha, beta)
    using a gibbs sampler.

    Parameters
    ----------
    y : ndarray of shape (N,)
        The data
    mu0 : float
        The prior mean parameter for mu
    sigma02 : float > 0
        The prior variance parameter for mu
    alpha : float > 0
        The prior alpha parameter for sigma2
    beta : float > 0
        The prior beta parameter for sigma2
    n_samples : int
        The number of samples to draw

    Returns
    -------
    samples : ndarray of shape (n_samples,2)
        1st col = mu samples, 2nd col = sigma2 samples
    """
    # initialization
    samples = np.empty((n_samples, 2))
    N = len(y)
    mu = y.mean()
    sigma2 = 25.
    # initialize posterior alpha, since it doesn't depend on mu or sigma2
    alphastar = alpha + N/2.
    for k in xrange(n_samples):
        # get the posterior parameters and draw mu
        sigstar2 = 1./((1./sigma02) + (N/sigma2))
        mustar = sigstar2*((mu0/sigma02) + y.sum()/sigma2)
        mu = st.norm.rvs(mustar, scale = sqrt(sigstar2))


        # get posterior parameters and draw sigma2
        betastar = beta + ((y-mu)**2).sum()/2.
        sigma2 = st.invgamma.rvs(alphastar, scale=betastar)

        # save sample
        samples[k,0] = mu
        samples[k,1] = sigma2
    return samples
    
# code for plotting KDEs of posteriors, and for getting posterior predictive is contained in plots.py

# below is the LDA solutions
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
        self.nzw = sp.zeros((self.n_topics,self.n_words))
        self.nmz = sp.zeros((self.n_docs,self.n_topics))
        self.nz = sp.zeros(self.n_topics)
        self.topics = {}
        random_distribution = sp.ones(self.n_topics)/float(self.n_topics)
        for m in xrange(self.n_docs):
            for i in self.documents[m]:
                z = sp.random.multinomial(1,random_distribution).argmax()
                self.nzw[z,self.documents[m][i]] += 1
                self.nmz[m,z] += 1
                self.nz[z] += 1
                self.topics[(m,i)] = z

    def sample(self,filename, burnin=100, sample_rate=10, n_samples=10, stopwords=None):
        self.buildCorpus(filename,stopwords)
        self.initialize()
        self.total_nzw = sp.zeros((self.n_topics,self.n_words))
        self.total_nmz = sp.zeros((self.n_docs,self.n_topics))
        self.logprobs = sp.zeros(burnin + sample_rate*n_samples)
        for i in xrange(burnin):
            self._sweep()
            self.logprobs[i] = self._loglikelihood()
            print "Iteration: {}".format(i) + "\tLog-prob: {}".format(self.logprobs[i])
        for i in xrange(n_samples*sample_rate):
            self._sweep()
            self.logprobs[i+burnin] = self._loglikelihood()
            print "Iteration: {}".format(i+burnin) + "\tLog-prob: {}".format(self.logprobs[i+burnin])
            if not i%sample_rate:
                self.total_nzw += sp.copy(self.nzw)
                self.total_nmz += sp.copy(self.nmz)

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
        """ This function is given. """
        dist = (self.nmz[m,:] + self.alpha) * (self.nzw[:,w] + self.beta) / (self.nz + self.beta*self.n_words)
        return dist/sum(dist)

    def _sweep(self):
        for m in xrange(self.n_docs):
            for i in self.documents[m]:
                w = self.documents[m][i]
                z = self.topics[(m,i)]
                self.nzw[z,w] -= 1
                self.nmz[m,z] -= 1
                self.nz[z] -= 1
                
                z = sp.random.multinomial(1,self._conditional(m,w)).argmax()

                self.nzw[z,w] += 1
                self.nmz[m,z] += 1
                self.nz[z] += 1
                self.topics[(m,i)] = z

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

