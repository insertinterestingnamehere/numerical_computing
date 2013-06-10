import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt

def initialize(n):
	p=0.5
	state = stats.bernoulli.rvs(p,size=(n**2)).astype(float)
	s = sum(state==0)
	state[state==0] = -sp.ones(s)
	return state.reshape(n,n)

#spinconfig = initialize(100)
#plt.imshow(spinconfig)
#plt.show()

def computeEnergy(state):
	n = state.shape[0]
	energy = 0
	for i in xrange(n):
		for j in xrange(n):
			energy -= state[i,j]*(state[i-1,j] + state[i,j-1])
	return energy

def propose(state):
	n = state.shape[0]
	val = sp.random.multinomial(1,sp.ones(n**2)/(n**2)).reshape(n,n)
	index = sp.argmax(val)
	i = index/n
	j = index%n
	return i,j

def proposedEnergy(i,j,energy,state):
	n = state.shape[0]
	S = state[i,j]
	temp = state[(i-1)%n,j] + state[(i+1)%n,j] + state[i,(j-1)%n] + state[i,(j+1)%n]
	return energy + 2*S*temp

def accept(energy,newEnergy,beta=10):
	if newEnergy <= energy:
		accepted = True
	elif stats.bernoulli.rvs(sp.exp(-beta*(newEnergy - energy)),size=1)[0] == 1:
		accepted = True
	else:
		accepted = False
	return accepted

def mcmc(n,beta,burnin=100000,n_samples=5000):
	state = initialize(n)
	energy = computeEnergy(state)
	logprobs = sp.zeros(burnin + n_samples)
	for k in xrange(burnin):
		print k
		logprobs[k] = -beta * energy
		i,j = propose(state)
		newEnergy = proposedEnergy(i,j,energy,state)
		accepted = accept(energy,newEnergy,beta)
		if accepted:
			state[i,j] *= -1
			energy = newEnergy
	samples = []
	for k in xrange(n_samples):
		logprobs[k + burnin] = -beta * energy
		i,j = propose(state)
		newEnergy = proposedEnergy(i,j,energy,state)
		accepted = accept(energy,newEnergy,beta)
		if accepted:
			state[i,j] *= -1
			energy = newEnergy
		if not k%100:
			samples.append(sp.copy(state))
	return samples,logprobs
