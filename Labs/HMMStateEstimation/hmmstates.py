import scipy as sp
import string
from scipy import linalg as la

def initialize(N,M):
	A = sp.ones((N,N))/N + sp.random.uniform(-1./N,1./N,N*N).reshape(N,N)
	A = (A.T/sp.sum(A,1)).T
	B = sp.ones((N,M))/M + sp.random.uniform(-1./M,1./M,N*M).reshape(N,M)
	B = (B.T/sp.sum(B,1)).T
	pi = sp.ones(N)/N + sp.random.uniform(-1./N,1./N,N)
	pi /= sp.sum(pi)
	return [A,B,pi]

def generate(hmm,n_sim):
	A = hmm[0]
	B = hmm[1]
	pi = hmm[2]
	observation_space = hmm[3]
	states = sp.zeros(n_sim)
	observations = []
	states[0] = sp.argmax(sp.random.multinomial(1,pi))
	observations.append(observation_space[sp.argmax(sp.random.multinomial(1,B[states[0],:]))])
	for i in range(1,n_sim):
		states[i] = sp.argmax(sp.random.multinomial(1,A[states[i-1],:]))
		observations.append(observation_space[sp.argmax(sp.random.multinomial(1,B[states[i],:]))])
	return states,observations

def transformObservations(observation_space,observations):
	M = len(observation_space)
	T = len(observations)
	obs = sp.zeros(T)
	for i in xrange(M):
		obs += sp.array([observations[j] == observation_space[i] for j in xrange(T)])*i
	return(obs)

def alphaPass(hmm,obs):
	A = hmm[0]
	B = hmm[1]
	pi = hmm[2]
	N = A.shape[0]
	T = len(obs)
	alpha = sp.zeros((T,N))
	C = sp.zeros(T)
	alpha[0,:] = pi*B[:,obs[0]]
	if sum(alpha[0,:]) == 0:
		return alpha,C,float('-inf')
	C[0] = 1/sum(alpha[0,:])
	alpha[0,:] = C[0]*alpha[0,:]
	for t in range(1,T):
		alpha[t,:] = sp.dot(alpha[t-1,:],A)*B[:,obs[t]]
		if sum(alpha[t,:]) == 0:
			return alpha,C,float('-inf')
		C[t] = 1/sum(alpha[t,:])
		alpha[t,:] = C[t]*alpha[t,:]
	return alpha,C,-sum(sp.log(C))

def betaPass(hmm,obs):
	A = hmm[0]
	B = hmm[1]
	pi = hmm[2]
	N = A.shape[0]
	alpha,C,logProb = alphaPass(hmm,obs)
	T = len(obs)
	beta = sp.ones((T,N))
	beta[T-1,:] = [C[T-1]]*N
	for t in range(T-2,-1,-1):
		beta[t,:] = C[t]*sp.dot(A,B[:,obs[t+1]]*beta[t+1,:])
	return(alpha,beta,logProb)

def gamma(alpha,beta):
	return(((alpha*beta).T/sp.sum(alpha*beta,1)).T)

def stateEstimation(hmm,observations):
	obs = transformObservations(hmm[3],observations)
	alpha,beta,logProb = betaPass(hmm,obs)
	gam = gamma(alpha,beta)
	return(sp.argmax(gam,1))

def misclassificationRate(trueStates,stateEstimates):
	T = len(trueStates)
	return(sum(trueStates - stateEstimates != 0)/float(T))
