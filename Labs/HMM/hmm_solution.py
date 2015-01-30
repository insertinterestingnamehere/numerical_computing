import numpy as np
from matplotlib import pyplot as plt
import string

class hmm(object):
    """
    Finite state space hidden markov model.
    """
    def __init__(self):
        """
        Initialize model parameters.

        Parameters
        ----------
        A : ndarray of shape (n,n)
            Column-stochastic state transition matrix.
        B : ndarray of shape (m,n)
            Column-stochastic observation matrix
        pi : ndarray of shape (n,)
            Initial state distribution
        """
        self.A = None
        self.B = None
        self.pi = None

    def _log_prob(self, c):
        """
        Calculate the probability of an observation sequence given model parameters,
        using the output of the forward pass.

        Parameters
        ----------
        c : ndarray of shape (T,)
            The scaling numbers from forward pass

        Returns
        -------
        out : float
            The log probability of obs given model parameters
        """
        return -(np.log(c)).sum()

    def _forward(self, obs):
        T = len(obs)
        n = self.A.shape[0]
        alpha = np.zeros((T,n))
        c = np.zeros(T)

        alpha[0,:] = self.pi*self.B[obs[0],:]
        c[0] = 1./(alpha[0,:].sum())
        alpha[0,:] *= c[0]

        for i in xrange(1,T):
            alpha[i,:] = (self.A.dot(alpha[i-1,:]))*self.B[obs[i],:]
            c[i] = 1./(alpha[i,:].sum())
            alpha[i,:] *= c[i]

        return alpha, c

    def _backward(self, obs, c):
        T = len(obs)
        n = self.A.shape[0]
        beta = np.zeros((T,n))

        beta[-1,:] = c[-1]
        for i in xrange(T-2,-1,-1):
            beta[i,:] = c[i]*((self.A.T).dot(self.B[obs[i+1],:]*beta[i+1,:]))

        return beta

    def _delta(self, obs, alpha, beta):
        T, n = alpha.shape
        delta = np.zeros((T-1,n,n))
        gamma = np.zeros((T,n))
        for t in xrange(T-1):
            delta[t,:,:] = (self.B[obs[t+1],:]*beta[t+1,:])*(self.A[:,:].T)*alpha[t,:].reshape((n,1))
            delta[t,:,:] /= delta[t,:,:].sum()
        gamma[:-1,:] = delta.sum(axis=2)
        gamma[-1,:] = alpha[-1,:]*beta[-1,:]/(alpha[-1,:]*beta[-1,:]).sum()
        return delta, gamma

    def _estimate(self, obs, delta, gamma):
        self.pi = gamma[0,:]
        self.A = delta.sum(axis=0).T/gamma[:-1,:].sum(axis=0)
        for j in xrange(self.B.shape[0]):
            self.B[j,:] = gamma[obs==j].sum(axis=0)
        self.B /= gamma.sum(axis=0)

    def fit(self, obs, A, B, pi, max_iter=100, tol=1e-3):
        """
        Use EM to fit model parameters to a given observation sequence.

        Parameters
        ----------
        obs : ndarray of shape (T,)
            Observation sequence on which to train the model.
        A : stochastic ndarray of shape (N,N)
            Initialization of state transition matrix
        B : stochastic ndarray of shape (M,N)
            Initialization of state-observation matrix
        pi : stochastic ndarray of shape (N,)
            Initialization of initial state distribution
        max_iter : integer
            The maximum number of iterations to take
        tol : float
            The convergence threshold for change in log-probability
        """

        self.A = A.copy()
        self.B = B.copy()
        self.pi = pi.copy()

        old_ll = 0
        ll = 0
        log_liks = []

        for i in xrange(max_iter):
            alpha, c = self._forward(obs)
            beta = self._backward(obs, c)
            delta, gam = self._delta(obs, alpha, beta)
            self._estimate(obs, delta, gam)
            ll = -np.log(c).sum()
            log_liks.append(ll)
            if abs(ll-old_ll) < tol:
                break
            else:
                old_ll = ll

def trainHMM():
    # load and process the data
    with open("declaration.txt", 'r') as f:
        dec = f.read(-1).lower()
    dec = dec.translate(string.maketrans("",""), string.punctuation+"\n")
    char_map = list(set(dec))
    obs = []
    for i in dec:
        obs.append(char_map.index(i))
    obs = np.array(obs)
    # train the HMM on the declaration dataset
    N = 2
    M = len(char_map)
    A = np.random.dirichlet(np.ones(N), size=N).T
    B = np.random.dirichlet(np.ones(M), size=N).T
    pi = np.random.dirichlet(np.ones(N))
    h = hmm()
    h.fit(obs,A, B, pi, max_iter=200)
    return h
