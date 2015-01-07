import scipy as sp
from scipy import stats as stats
from scipy import linalg as la

def posterior(prior,depthProbs,x,y):
	posterior = sp.zeros((20,20))
	for i in xrange(20):
		for j in xrange(20):
			posterior[i,j] = prior[i,j]/(1-prior[i,j]*depthProbs[i,j])
	posterior[x,y] = (prior[x,y]*(1-depthProbs[x,y]))/((1-prior[x,y]) + prior[x,y]*(1-depthProbs[x,y]))
	return(posterior/sp.sum(posterior))

def nextTarget(p):
	ind = sp.argmax(p)
	x = ind/20
	y = ind%20
	return x,y

def search(location,prior,depthProbs):
	found = False
	n_iters = 0
	while found == False:
		n_iters += 1
		x,y = nextTarget(prior*depthProbs)
		if (x==location[0] and y==location[1]):
			success = stats.bernoulli.rvs(depthProbs[x,y])
			if success:
				found = True
		prior = posterior(prior,depthProbs,x,y)
	return n_iters,sp.array([x,y])

def searchSimulation(location,prior,depthProbs,n_sims):
	searches = sp.zeros(n_sims)
	for i in xrange(n_sims):
		searches[i] = search(location,prior,depthProbs)[0]
	
	print "Shortest Search Length: {}".format(searches.min())
	print "Longest Search Length: {}".format(searches.max())
	print "Average Search Length: {}".format(sp.mean(searches))
	return searches
