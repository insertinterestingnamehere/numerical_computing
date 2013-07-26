#================================================
#Plots for the Value Function Iteration Lab
#================================================
import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import matplotlib.pyplot as plt
import scipy as sp

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
