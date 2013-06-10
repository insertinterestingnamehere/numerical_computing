import scipy as sp
import scipy.linalg as la
from scipy.stats import bernoulli

def acceptance(x,y,mu,sigma):
	p = min(1,sp.exp(-0.5 * (sp.dot(x-mu,la.solve(sigma,x-mu)) - sp.dot(y - mu, la.solve(sigma,y-mu)))))
	return bernoulli.rvs(p)

def nextState(x,mu,sigma):
	K = len(x)
	y = sp.random.multivariate_normal(x,sp.eye(K))
	accept = acceptance(y,x,mu,sigma)
	if accept:
		return y
	else:
		return x

def lmvnorm(x,mu,sigma):
	return -0.5 * (sp.dot(x-mu,la.solve(sigma,x-mu)) - len(x)*sp.log(2*sp.pi) - sp.log(la.det(sigma)))

def metropolis(x,mu,sigma,n_samples=1000):
	logprobs = sp.zeros(n_samples)
	x_samples = sp.zeros((n_samples,len(x)))
	for i in xrange(n_samples):
		logprobs[i] = lmvnorm(x,mu,sigma)
		x = nextState(x,mu,sigma)
		x_samples[i,:] = sp.copy(x)
	return x_samples,logprobs
