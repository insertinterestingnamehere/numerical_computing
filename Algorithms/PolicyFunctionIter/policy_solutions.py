#Solutions to Policy Function Iteration Lab

import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg
import math
from matplotlib import pyplot as plt
from scipy import linalg as la


def u(x):
    return np.sqrt(x).flatten()
    
def policyIter(beta, N, Wmax=1.):
    """
    Solve the infinite horizon cake eating problem using policy function iteration.
    Inputs:
        beta -- float, the discount factor
        N -- integer, size of discrete approximation of cake
        Wmax -- total amount of cake available
    Returns:
        values -- converged value function (Numpy array of length N)
        psi -- converged policy function (Numpy array of length N)
    """
    W = np.linspace(0,Wmax,N) #state space vector
    I = sparse.identity(N, format='csr')
    
    #precompute u(W-W') for all possible inputs
    actions = np.tile(W, N).reshape((N,N)).T
    actions = actions - actions.T
    actions[actions<0] = 0
    rewards = np.sqrt(actions)
    rewards[np.triu_indices(N, k=1)] = -1e10 #pre-computed reward function
    
    psi_ind = np.arange(N)
    rows = np.arange(0,N)
    tol = 1.
    while tol >= 1e-9:
        columns = psi_ind
        data = np.ones(N)
        Q = sparse.coo_matrix((data,(rows,columns)),shape=(N,N))
        Q = Q.tocsr()
        values = linalg.spsolve(I-beta*Q, u(W-W[psi_ind])).reshape(1,N)
        psi_ind1 = np.argmax(rewards + beta*values, axis=1)
        tol = math.sqrt(((W[psi_ind] - W[psi_ind1])**2).sum())
        psi_ind = psi_ind1
    return values.flatten(), W[psi_ind]

def modPolicyIter(beta, N, Wmax=1., m=15):
    """
    Solve the infinite horizon cake eating problem using modified policy function iteration.
    Inputs:
        beta -- float, the discount factor
        N -- integer, size of discrete approximation of cake
        Wmax -- total amount of cake available
    Returns:
        values -- converged value function (Numpy array of length N)
        psi -- converged policy function (Numpy array of length N)
    """
    W = np.linspace(0,Wmax,N) #state space vector
    
    #precompute u(W-W') for all possible inputs
    actions = np.tile(W, N).reshape((N,N)).T
    actions = actions - actions.T
    actions[actions<0] = 0
    rewards = np.sqrt(actions)
    rewards[np.triu_indices(N, k=1)] = -1e10 #pre-computed reward function
    
    psi_ind = np.arange(N)
    values = np.zeros(N)
    tol = 1.
    while tol >= 1e-9:
        for i in xrange(m):
            values = u(W - W[psi_ind]) + beta*values[psi_ind]
        psi_ind1 = np.argmax(rewards + beta*values.reshape(1,N), axis=1)
        tol = math.sqrt(((W[psi_ind] - W[psi_ind1])**2).sum())
        psi_ind = psi_ind1
    return values.flatten(), W[psi_ind]

