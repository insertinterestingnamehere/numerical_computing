import numpy as np
from scipy import random as rand
from scipy import linalg as la


def generation(Fk, Q, U, H, R, x_initial, n_iters):
    N = Fk.shape[0]
    states = np.zeros((N, n_iters))
    states[:,0] = x_initial
    observations = np.zeros((H.shape[0], n_iters))
    observations[:,0] = x_initial[0:2]
    mean = np.zeros(H.shape[0])
    for i in xrange(n_iters - 1):
        states[:,i+1] = np.dot(Fk, states[:,i]) + \
            U + rand.multivariate_normal(np.zeros(N), Q, 1)
        observations[:,i+1] = np.dot(H, states[:,i+1]) + rand.multivariate_normal(mean, R, 1)
    return states, observations


def kalmanFilter(Fk, Q, U, H, R, x_initial, P, observations):
    T = observations.shape[1]
    N = Fk.shape[0]
    estimated_states = np.zeros((N, T))
    for i in xrange(T):
        if i == 0:
            estimated_states[:,i] = np.dot(Fk, x_initial) + U
        else:
            estimated_states[:,i] = np.dot(Fk, estimated_states[:,i-1]) + U
        P = np.dot(np.dot(Fk, P), Fk.T) + Q
        yk = observations[:,i] - np.dot(H, estimated_states[:,i])
        S = np.dot(np.dot(H, P), H.T) + R
        K = np.dot(np.dot(P, H.T), la.inv(S))
        estimated_states[:,i] = estimated_states[:,i] + np.dot(K, yk)
        P = np.dot((np.eye(N) - np.dot(K, H)), P)
    return estimated_states


def predict(Fk, U, x_initial, T):
    N = Fk.shape[0]
    predicted_states = np.zeros((N, T))
    for i in xrange(T):
        if i == 0:
            predicted_states[:,i] = np.dot(Fk, x_initial) + U
        else:
            predicted_states[:,i] = np.dot(Fk, predicted_states[:,i-1]) + U
    return predicted_states[0:2,:]


def rewind(Fk, U, x_initial, T):
    N = Fk.shape[0]
    rewound_states = np.zeros((N, T))
    Fk = la.inv(Fk)
    U = -np.dot(Fk, U)
    for i in xrange(T):
        if i == 0:
            rewound_states[:,i] = np.dot(Fk, x_initial) + U
        else:
            rewound_states[:,i] = np.dot(Fk, rewound_states[:,i-1]) + U
    return rewound_states[0:2,:]
