#================================================
#Plots for the Value Function Iteration Lab
#================================================
import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
import numpy as np
import math
from scipy import stats as st
import discretenorm
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


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
            ax1.plot_surface(states[X],Y,values.T, cmap=cm.coolwarm)
            plt.show ()
            
            fig2 = plt.figure() 
            ax2 = Axes3D(fig2)
            y = np.arange(0,T+1)
            X,Y=np.meshgrid(x,y)
            ax2.plot_surface(states[X],Y,psi.T, cmap = cm.coolwarm)
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

def finite_horiz():
    #First compute solution to problem 1
    beta = 0.9;
    T = 10;
    N = 100;
    u = lambda c: np.sqrt(c);
    W = np.linspace(0,1,N);
    X, Y = np.meshgrid(W,W);
    Wdiff = Y-X
    index = Wdiff <0;
    Wdiff[index] = 0;
    util_grid = u(Wdiff);
    util_grid[index] = -10**10;
    V = np.zeros((N,T+2));
    psi = np.zeros((N,T+1));
    
    
    for k in xrange(T,-1,-1):
        val = util_grid + beta*np.tile(V[:,k+1].T,(N,1));
        vt = np.amax(val, axis = 1);
        psi_ind = np.argmax(val,axis = 1)
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
    plt.clf()
    
#plot delta -> 0    
def delta():     
    beta = 0.99
    N = 1000
    u = lambda c: np.sqrt(c)
    W = np.linspace(0,1,N)
    X, Y = np.meshgrid(W,W)
    Wdiff = (X-Y).T
    index = Wdiff <0
    Wdiff[index] = 0
    util_grid = u(Wdiff)
    util_grid[index] = -10**10
    
    Vprime = np.zeros((N,1))
    delta = np.ones(1)
    tol = 10**-9
    it = 0
    max_iter = 500
    
    while (delta[-1] >= tol) and (it < max_iter):
        V = Vprime
        it += 1;
        val = util_grid + beta*V.T
        Vprime = np.amax(val, axis = 1)
        Vprime = Vprime.reshape((N,1))
        delta = np.append(delta,np.dot((Vprime-V).T,Vprime-V))
        
    plt.figure()
    plt.plot(delta[1:])
    plt.ylabel(r'$\delta_k$')
    plt.xlabel('iteration')
    plt.savefig('convergence.pdf')
    plt.clf()

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

def disc_norm():
    x = np.linspace(-3,3,100)
    y = st.norm.pdf(x,0,1)
    fig, ax = plt.subplots()
    fig.canvas.draw()
    
    ax.plot(x,y)
    
    fill1_x = np.linspace(-2,-1.5,100)
    fill1_y = st.norm.pdf(fill1_x,0,1)
    fill2_x = np.linspace(-1.5,-1,100)
    fill2_y = st.norm.pdf(fill2_x,0,1)
    ax.fill_between(fill1_x,0,fill1_y,facecolor = 'blue', edgecolor = 'k',alpha = 0.75)
    ax.fill_between(fill2_x,0,fill2_y,facecolor = 'blue', edgecolor = 'k',alpha = 0.75)
    for label in ax.get_yticklabels():
        label.set_visible(False)
    for tick in ax.get_xticklines():
        tick.set_visible(False)
    for tick in ax.get_yticklines():
        tick.set_visible(False)
    
    plt.rc("font", size = 16)
    plt.xticks([-2,-1.5,-1])
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[0] = r"$v_k$"
    labels[1] = r"$\varepsilon_k$"
    labels[2] = r"$v_{k+1}$"
    ax.set_xticklabels(labels)
    plt.ylim([0, .45])

    
    plt.savefig('discnorm.pdf')
    plt.clf()
    
def stoch_value():    
    #Compute Solution==========================================================
    sigma = .5
    mu = 4*sigma
    K = 7
    Gamma, eps = discretenorm.discretenorm(K,mu,sigma)
    
    N = 100
    W = np.linspace(0,1,N)
    V = np.zeros((N,K))
    
    u = lambda c: np.sqrt(c)
    beta = 0.99
    
    X,Y= np.meshgrid(W,W)
    Wdiff = Y-X
    index = Wdiff < 0
    Wdiff[index] = 0
    
    util_grid = u(Wdiff)
    
    util3 = np.tile(util_grid[:,:,np.newaxis],(1,1,K))
    eps_grid = eps[np.newaxis,np.newaxis,:]
    eps_util = eps_grid*util3
    
    Gamma_grid = Gamma[np.newaxis,:]
    
    delta = 1
    Vprime = V
    z = 0
    while (delta > 10**-9):
        z= z+1
        V = Vprime
        gamV = Gamma_grid*V
        Expval = np.sum(gamV,1)
        Exp_grid = np.tile(Expval[np.newaxis,:,np.newaxis],(N,1,K))
        arg = eps_util+beta*Exp_grid
        arg[index] = -10^10
        Vprime = np.amax(arg,1)
        psi_ind = np.argmax(arg,1)
        psi = W[psi_ind]
        delta = np.linalg.norm(Vprime - V)
    
    #============================================================    
    #Plot 3D    
    x=np.arange(0,N)
    y=np.arange(0,K)
    X,Y=np.meshgrid(x,y)
    fig1 = plt.figure()
    ax1= Axes3D(fig1)
    ax1.set_xlabel(r'$W$')
    ax1.set_ylabel(r'$\varepsilon$')
    ax1.set_zlabel(r'$V$')
    ax1.plot_surface(W[X],Y,np.transpose(Vprime), cmap=cm.coolwarm)
    plt.savefig('stoch_value.pdf')
    plt.clf()
    
    
if __name__ == "__main__":
    disc_norm()
    stoch_value()
    finite_horiz()
    delta()
    infiniteHorizon()
