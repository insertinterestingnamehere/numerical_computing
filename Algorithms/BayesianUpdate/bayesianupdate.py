import scipy as sp
import math
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.stats import norm

trial = open('trial.csv','r')
data = sp.array([int(line) for line in trial])

def bernoulli_posterior(data,prior):
	n_1 = sum(data)
	n_2 = len(data) - n_1
	x = sp.arange(0,1.01,step=.01)
	y = beta.pdf(x,prior[0] + n_1 - 1, prior[1] + n_2 - 1)
	plt.plot(x,y)
	plt.show()
	return

bernoulli_posterior(data,sp.array([8.,2.]))

def multinomial_beta(alpha):
	return sp.prod(sp.array([math.gamma(alpha[i]) for i in xrange(len(alpha))]))/math.gamma(sp.sum(alpha))

fruit = open('fruit.csv','r')
data = sp.array([line.rstrip() for line in fruit])

def multinomial_posterior(data,prior):
	categories = sp.sort(list(set(data)))
	n = len(categories)
	sums = sp.zeros(n)
	for i in xrange(n):
		sums[i] = sum(data==categories[i])
	return prior + sums

multinomial_posterior(data,sp.array([5.,3.,4.,2.,1.]))

def invgammapdf(x,alpha,beta):
	alpha = float(alpha)
	beta = float(beta)
	if (type(x) != int) & (type(x) != float):
		return (beta**alpha / math.gamma(alpha))*sp.array([(x[i]**(-alpha - 1))*math.exp(-beta/x[i]) for i in xrange(len(x))])
	else:
		return (beta**alpha / math.gamma(alpha))*(x**(-alpha - 1))*math.exp(-beta/x)

def mean_posterior(data,prior,variance):
	n = float(len(data))
	sample_mean = sp.mean(data)
	mu_0 = prior[0]
	sigma2_0 = prior[1]
	posterior_mean = ((mu_0*variance)/n + sample_mean*sigma2_0)/(variance/n + sigma2_0)
	posterior_variance = 1/(n/variance + 1/sigma2_0)
	return posterior_mean,posterior_variance

scores = open('examscores.csv','r')
data = sp.array([float(line) for line in scores])

posterior_mean, posterior_variance = mean_posterior(data,sp.array([74.,25.]),36.)
x = sp.arange(0,100.1,step=0.1)
y = norm.pdf(x,loc=posterior_mean,scale=sp.sqrt(posterior_variance))
plt.plot(x,y)
plt.show()

def variance_posterior(data,prior,mean):
	n = float(len(data))
	alpha = prior[0]
	beta = prior[1]
	posterior_alpha = alpha + n/2.
	posterior_beta = beta + sp.dot(data-mean,data-mean)/2.
	return posterior_alpha, posterior_beta

posterior_alpha, posterior_beta = variance_posterior(data,sp.array([2.,25.]),62.)
x = sp.arange(0.1,100.1,step=0.1)
y = invgammapdf(x,posterior_alpha,posterior_beta)
plt.plot(x,y)
plt.show()
