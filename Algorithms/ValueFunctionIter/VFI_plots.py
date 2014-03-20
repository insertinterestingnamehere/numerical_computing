#================================================
#Plots for the Value Function Iteration Lab
#================================================
import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
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
    n_range = np.arange(N) #this is used several times, so initialize it once
    if finite:
        values = np.zeros((N, T+2))
        psi = np.zeros((N, T+1))
        for i in xrange(T,-1,-1):
            argmaxs = np.argmax(rewards + beta*values[:,i+1].reshape(1,N), axis=1)
            values[:,i] = (rewards + beta*values[:,i+1].reshape(1,N))[n_range,argmaxs]
            psi[:,i] = states[argmaxs]
            x=np.arange(0,N)
        
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

def plot_finite_horiz():
    #First compute solution to problem 1
    beta = 0.9;
    T = 10;
    N = 100;
    u = lambda c: sp.sqrt(c);
    W = sp.linspace(0,1,N);
    X, Y = sp.meshgrid(W,W);
    Wdiff = Y-X
    index = Wdiff <0;
    Wdiff[index] = 0;
    util_grid = u(Wdiff);
    util_grid[index] = -10**10;
    V = sp.zeros((N,T+2));
    psi = sp.zeros((N,T+1));
    
    
    for k in xrange(T,-1,-1):
        val = util_grid + beta*sp.tile(sp.transpose(V[:,k+1]),(N,1));
        vt = sp.amax(val, axis = 1);
        psi_ind = sp.argmax(val,axis = 1)
        V[:,k]    = vt;
        psi[:,k]    = W[psi_ind];
    
    #now create plots
    #fixed time plot
    
    plt.figure()
    plt.plot(V[:,5])
    plt.title(r'Value function for $t = 5$')
    plt.ylabel(r'$V$')
    plt.xlabel(r'$W$')
    plt.savefig('fixed_time.pdf')   
    
    #fixed W plot
    plt.figure()
    plt.plot(V[50,:])
    plt.title(r'Value function for $W = 0.505$')
    plt.ylabel(r'$V$')
    plt.xlabel(r'$t$')
    plt.savefig('fixed_w.pdf')
    
#plot delta -> 0    
def plot_delta():     
    beta = 0.99
    N = 1000
    u = lambda c: sp.sqrt(c)
    W = sp.linspace(0,1,N)
    X, Y = sp.meshgrid(W,W)
    Wdiff = sp.transpose(X-Y)
    index = Wdiff <0
    Wdiff[index] = 0
    util_grid = u(Wdiff)
    util_grid[index] = -10**10
    
    Vprime = sp.zeros((N,1))
    delta = sp.ones(1)
    tol = 10**-9
    it = 0
    max_iter = 500
    
    while (delta[-1] >= tol) and (it < max_iter):
        V = Vprime
        it += 1;
        print(it)
        val = util_grid + beta*sp.transpose(V)
        Vprime = sp.amax(val, axis = 1)
        Vprime = Vprime.reshape((N,1))
        delta = sp.append(delta,sp.dot(sp.transpose(Vprime - V),Vprime-V))
        
    plt.figure()
    plt.plot(delta[1:])
    plt.ylabel(r'$\delta_k$')
    plt.xlabel('iteration')
    plt.savefig('convergence.pdf')

def infiniteHorizon():
    """
    Plot policy function for infinite time horizon cake eating problem.
    """
    values, psi = eatCake(.9, 100, finite=False)
    states = np.linspace(0,1,100)
    plt.figure()
    plt.title(r'Policy Function')
    plt.ylabel(r'$\psi$')
    plt.xlabel(r'$W$')
    plt.plot(states, psi)
    plt.savefig('infiniteHorizon.pdf')
    plt.clf()
infiniteHorizon()
