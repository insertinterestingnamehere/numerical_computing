# Formatted Better than the other solutions file

import numpy as np
import math
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits . mplot3d import Axes3D

def eatCake(beta, N, Wmax=1., T=None, finite=True, plot=False):
    """
    Solve the finite horizon cake-eating problem using Value Function iteration.
    Inputs:
        T -- final time period
        beta -- discount factor
        N -- number of discrete intervals to break up the cake
        size -- size of the cake to begin with
        plot -- boolean indicating whether to plot value function surface and policy function
                surface
    Returns:
        values -- numpy array of shape (N, T+2) (if finite=True) or shape (N,) (if finite=False)
                  giving the value function at each time period for each state
        psi -- numpy array of shape (N, T+1) (if finite=True) or shape (N,) (if finite=False)
               giving the policy at each time period for each state.
    """
    states = np.linspace(0,Wmax,N) #state space vector
    actions = np.tile(states, N).reshape((N,N)).T
    actions = actions - actions.T
    actions[actions<0] = 0
    rewards = np.sqrt(actions)
    rewards[np.triu_indices(N, k=1)] = -1e10 #pre-computed reward function
    print rewards
    n_range = np.arange(N) #this is used several times, so initialize it once
    if finite:
        values = np.zeros((N, T+2))
        psi = np.zeros((N, T+1))
        for i in xrange(T,-1,-1):
            argmaxs = np.argmax(rewards + beta*values[:,i+1].reshape(1,N), axis=1)
            values[:,i] = (rewards + beta*values[:,i+1].reshape(1,N))[n_range,argmaxs]
            psi[:,i] = states[argmaxs]
        
        if plot:
            x=np.arange(0,N)
            y=np.arange(0,T+2)
            X,Y=np.meshgrid(x,y)
            fig1 = plt.figure()
            ax1= Axes3D(fig1)
            ax1.plot_surface(states[X],Y,sp.transpose(values), cmap=cm.coolwarm)
            plt.show ()
            
            fig2 = plt.figure() 
            ax2 = Axes3D(fig2)
            y = np.arange(0,T+1)
            X,Y=np.meshgrid(x,y)
            ax2.plot_surface(states[X],Y,sp.transpose(psi), cmap = cm.coolwarm)
            plt.show()
    else:
        values = np.zeros(N)
        psi = np.zeros(N)
        delta = 1.
        while delta >= 1e-9:
            values1 = values.copy()
            argmaxs = np.argmax(rewards + beta*values1.reshape(1,N), axis=1)
            values = (rewards + beta*values.reshape(1,N))[n_range, argmaxs]
            psi = states[argmaxs]
            delta = ((values-values1)**2).sum()
        if plot:
            plt.plot(states, psi)
            plt.show()
            
    return values, psi

#Problem 1
values, psi = eatCake(.9, 100, T = 10, plot=True)

#Problem 2
values, psi = eatCake(.9, 100, T=1000)
plt.plot(psi[-1,:])
plt.show()

#Problem 3
values, psi = eatCake(.9, 100, finite=False, plot=True)

