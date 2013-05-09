import scipy as sp
from scipy import random as rand
from scipy import linalg as la

def generation(Fk,Q,U,H,R,x_initial,n_iters):
	N = Fk.shape[0]
	states = sp.zeros((N,n_iters))
	states[:,0] = x_initial
	observations = sp.zeros((H.shape[0],n_iters))
	observations[:,0] = x_initial[0:2]
	mean = sp.zeros(H.shape[0])
	for i in xrange(n_iters-1):
		states[:,i+1] = sp.dot(Fk,states[:,i]) + U + rand.multivariate_normal(sp.zeros(N),Q,1)
		observations[:,i+1] = sp.dot(H,states[:,i+1]) + rand.multivariate_normal(mean,R,1)
	return states,observations

def kalmanFilter(Fk,Q,U,H,R,x_initial,P,observations):
	T = observations.shape[1]
	N = Fk.shape[0]
	estimated_states = sp.zeros((N,T))
	for i in xrange(T):
		if i == 0:
			estimated_states[:,i] = sp.dot(Fk,x_initial) + U
		else:
			estimated_states[:,i] = sp.dot(Fk,estimated_states[:,i-1]) + U
		P = sp.dot(sp.dot(Fk,P),Fk.T) + Q
		yk = observations[:,i] - sp.dot(H,estimated_states[:,i])
		S = sp.dot(sp.dot(H,P),H.T) + R
		K = sp.dot(sp.dot(P,H.T),la.inv(S))
		estimated_states[:,i] = estimated_states[:,i] + sp.dot(K,yk)
		P = sp.dot((sp.eye(N) - sp.dot(K,H)),P)
	return estimated_states

def predict(Fk,U,x_initial,T):
	N = Fk.shape[0]
	predicted_states = sp.zeros((N,T))
	for i in xrange(T):
		if i == 0:
			predicted_states[:,i] = sp.dot(Fk,x_initial) + U
		else:
			predicted_states[:,i] = sp.dot(Fk,predicted_states[:,i-1]) + U
	return predicted_states[0:2,:]

def rewind(Fk,U,x_initial,T):
	N = Fk.shape[0]
	rewound_states = sp.zeros((N,T))
	Fk = la.inv(Fk)
	U = -sp.dot(Fk,U)
	for i in xrange(T):
		if i == 0:
			rewound_states[:,i] = sp.dot(Fk,x_initial) + U
		else:
			rewound_states[:,i] = sp.dot(Fk,rewound_states[:,i-1]) + U
	return rewound_states[0:2,:]
