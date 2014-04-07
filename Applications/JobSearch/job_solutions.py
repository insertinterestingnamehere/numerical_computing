import numpy as np
import math
from discretelognorm import discretelognorm
from matplotlib import pyplot as plt

#initialize parameters for both problems
m = 20
v = 200
N = 500
Wmax = 100
Wmin = 0
gamma = .10
alpha = .5
beta = .9
e_params = (m, v) #to be passed to discretelognorm


#Problem 1 Solution============================================================
def jobSearchVI(Wmin, Wmax, N, e_params, alpha, beta, gamma):
    """
    Solve the job search problem.
    VE denotes the employed value function.
    VU denotes the unemployed value function.
    EVU denotes E_{w''}V^U(w,w'').
    PSI denotes the policy function.
    Label employed with 0, unemployed with 1.
    """
    u   = lambda c: np.sqrt(c)
    w   = np.linspace(Wmin, Wmax, N)
    uaw = u(alpha*w).reshape((N,1))
    uw  = u(w)
    f   = discretelognorm(w, *e_params)
    
    VE   = np.zeros(N)
    EVU  = np.zeros(N)
    VU   = np.zeros((N,N))
    MVE  = np.empty((N,N)) #tiled version of VE
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
    plt.show()
    print "Number of iterations:", i
    return VE, VU, PSI

#Problem 2 Solution============================================================
def jobSearchMPI(Wmin, Wmax, N, e_params, alpha, beta, gamma):
    """
    Solve Job search problem using modified policy function iteration.
    """
    # initialize data
    u = lambda c: np.sqrt(c)
    w = np.linspace(Wmin, Wmax, N)
    uaw = u(alpha*w).reshape((N,1))
    uw = u(w)
    f = discretelognorm(w, *e_params)
    
    VE = np.zeros(N)
    EVU = np.zeros(N)
    VU = np.zeros((N,N))
    PSI = 2*np.ones((N,N)) #initialize policy function
    MVE = np.empty((N,N))
    MEVU = np.empty((N,N))
    
    delta = 10.
    it = 0
    while delta >= 1e-9:
        it += 1
        
        #calculate new policy function
        PSI1 = np.argmax(np.dstack([MEVU,MVE]), axis=2)
        
        #iterate on the value functions
        for i in xrange(15):
            MVE[:,:] = VE.reshape((1,N))
            MEVU[:,:] = EVU.reshape((N,1))
            VU = uaw + beta*(MVE*PSI1+MEVU*(1-PSI1))
            VE = uw + beta*((1-gamma)*VE + gamma*EVU)
            EVU = np.dot(VU,f).ravel()
            
        #test for convergence
        delta = math.sqrt(np.abs((PSI1-PSI)).sum())
        
        #update
        PSI = PSI1
        
    #calculate and plot reservation wage function
    wr_ind = np.argmax(np.diff(PSI), axis = 1)
    wr = w[wr_ind]
    plt.plot(w,wr)
    plt.show()
    print "Number of iterations:", it
    return PSI
