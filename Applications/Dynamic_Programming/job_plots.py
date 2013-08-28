import scipy as sp
import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import matplotlib.pyplot as plt
from discretelognorm import discretelognorm

#plot discrete policy function
def plot_disc_policy():
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
    
