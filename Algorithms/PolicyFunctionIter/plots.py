import scipy as sp
import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import matplotlib.pyplot as plt
import numpy as np
import math
from discretelognorm import discretelognorm

def reservation_wage():
    m = 20
    v = 200
    N = 500
    Wmax = 100
    Wmin = 0
    gamma = .10
    alpha = .5
    beta = .9
    e_params = (m, v)
    
    u = lambda c: np.sqrt(c)
    w = np.linspace(Wmin, Wmax, N)
    uaw = u(alpha*w).reshape((N,1))
    uw = u(w)
    f = discretelognorm(w, *e_params)
    
    VE = np.zeros(N)
    EVU = np.zeros(N)
    VU = np.zeros((N,N))
    MVE = np.empty((N,N)) #tiled version of VE
    MEVU = np.empty((N,N)) #tiled version of EVU
    
    delta = 1.
    i = 0
    while delta >= 1e-9:
        i+=1
        
        #update tiled value functions
        MVE[:,:] = VE.reshape((1,N))
        MEVU[:,:] = EVU.reshape((N,1))
        
        #calculate new value functions
        VU1 = uaw + beta*np.max(np.dstack([MEVU, MVE]), axis=2)
        VE1 = uw + beta*((1-gamma)*VE + gamma*EVU)
        
        #test for convergence
        d1 = ((VE1-VE)**2).sum()
        d2 = ((VU1-VU)**2).sum()
        delta = max(d1,d2)
        
        #update
        VU = VU1
        VE = VE1
        EVU = np.dot(VU,f).ravel()
    
    #calculate policy function
    PSI = np.argmax(np.dstack([MEVU,MVE]), axis=2)
    
    #calculate and plot reservation wage function
    wr_ind = np.argmax(np.diff(PSI), axis = 1)
    wr = w[wr_ind]
    plt.plot(w,wr)
    plt.savefig('reservation_wage.pdf')
    plt.clf()


#plot discrete policy function
def disc_policy():
    #First compute policy function...==========================================
    N = 500
    w = sp.linspace(0,100,N)
    w = w.reshape(N,1)
    u = lambda c: sp.sqrt(c)
    util_vec = u(w)
    alpha = 0.5
    alpha_util = u(alpha*w)
    alpha_util_grid = sp.repeat(alpha_util,N,1)
    
    m = 20
    v = 200
    f = discretelognorm(w,m,v)
    
    VEprime = sp.zeros((N,1))
    VUprime    = sp.zeros((N,N))
    EVUprime = sp.zeros((N,1))
    psiprime = sp.ones((N,1))
    gamma = 0.1
    beta = 0.9
    
    m = 15
    tol = 10**-9
    delta = 1+tol
    it = 0
    while (delta >= tol):
        it += 1
        
        psi = psiprime.copy()
        arg1 = sp.repeat(sp.transpose(VEprime),N,0)
        arg2 = sp.repeat(EVUprime,N,1)
        arg = sp.array([arg2,arg1])
        psiprime = sp.argmax(arg,axis = 0) 
        
        for j in sp.arange(0,m):
            VE = VEprime.copy()
            VU = VUprime.copy()
            EVU = EVUprime.copy()
            VEprime = util_vec + beta*((1-gamma)*VE + gamma*EVU)
            arg1 = sp.repeat(sp.transpose(VE),N,0)*psiprime
            arg2 = sp.repeat(EVU,N,1)*(1-psiprime)
            arg = arg1+arg2
            VUprime = alpha_util_grid + beta*arg
            EVUprime = sp.dot(VUprime,f)  
    
        
    
        delta = sp.linalg.norm(psiprime -psi) 

    wr_ind = sp.argmax(sp.diff(psiprime), axis = 1)
    wr = w[wr_ind]
    print w[250],wr[250]
        
    #Then plot=================================================================
    plt.plot(w,psiprime[250,:]) 
    plt.ylim([-.5,1.5])      
    plt.xlabel(r'$w\prime$')
    plt.yticks([0,1])
    plt.savefig('disc_policy.pdf')
    plt.clf()


if __name__ == "__main__":
    reservation_wage()
    disc_policy()