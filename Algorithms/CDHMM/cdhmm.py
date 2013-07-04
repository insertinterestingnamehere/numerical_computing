import scipy as sp
from scipy import linalg as la

def generateGaussianHMM(hmm,n_sim):
	A = hmm[0]
	means = hmm[1]
	covars = hmm[2]
	pi = hmm[3]
	states = sp.zeros(n_sim).astype(int)
	K = len(means[0,:])
	observations = sp.zeros((n_sim,K))
	states[0] = int(sp.argmax(sp.random.multinomial(1,pi)))
	observations[0,:] = sp.random.multivariate_normal(means[states[0],:],covars[states[0],:,:])
	for i in range(1,n_sim):
		states[i] = int(sp.argmax(sp.random.multinomial(1,A[states[i-1],:])))
		observations[i,:] = sp.random.multivariate_normal(means[states[i],:],covars[states[i],:,:])
	return states,observations

def generateGMMHMM(hmm,n_sim):
	A = hmm[0]
	components = hmm[1]
	means = hmm[2]
	covars = hmm[3]
	pi = hmm[4]
	states = sp.zeros(n_sim).astype(int)
	K = len(means[0,0,:])
	observations = sp.zeros((n_sim,K))
	comps = sp.zeros(n_sim).astype(int)
	states[0] = int(sp.argmax(sp.random.multinomial(1,pi)))
	comps[0] = int(sp.argmax(sp.random.multinomial(1,components[states[0],:])))
	observations[0,:] = sp.random.multivariate_normal(means[states[0],comps[0],:],covars[states[0],comps[0],:,:])
	for i in range(1,n_sim):
		states[i] = int(sp.argmax(sp.random.multinomial(1,A[states[i-1],:])))
		comps[i] = int(sp.argmax(sp.random.multinomial(1,components[states[i],:])))
		observations[i,:] = sp.random.multivariate_normal(means[states[i],comps[i],:],covars[states[i],comps[i],:,:])
	return states,comps,observations
