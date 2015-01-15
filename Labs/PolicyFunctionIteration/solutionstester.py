import scipy as sp
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot as plt

def Problem2Real():
    beta = 0.95
    N = 1000
    u = lambda c: sp.sqrt(c)
    psi_ind = sp.arange(0,N)

    W = sp.linspace(0,1,N)
    X, Y = sp.meshgrid(W,W)
    Wdiff = Y-X
    index = Wdiff <0
    Wdiff[index] = 0
    util_grid = u(Wdiff)

    I = sp.sparse.identity(N)
    delta = 1
    z = 0
    while (delta > 10**-9):
        z = z+1
        #print(z)
        psi_prev = psi_ind.copy()
    
        rows = sp.arange(0,N)
        columns = psi_ind
        data = sp.ones(N)
        Q = sp.sparse.coo_matrix((data,(rows,columns)),shape = (N,N))
        Q = Q.tocsr()
   
   #Solve for Value Function
        V = spsolve(I-beta*Q,u(W-W[psi_ind]))

    #Find Policy Function    
        arg = util_grid + beta*V
        arg[index] = -10**10
        psi_ind = sp.argmax(arg,axis = 1)
        delta = sp.amax(sp.absolute(W[psi_ind]-W[psi_prev]))


    return W[psi_ind]

def Problem3Real():
    beta = 0.95
    N = 1000
    u = lambda c: sp.sqrt(c)
    
    W = sp.linspace(0,1,N)
    W = W.reshape(N,1)
    X, Y = sp.meshgrid(W,W)
    Wdiff = Y-X
    index = Wdiff <0
    Wdiff[index] = 0
    util_grid = u(Wdiff)
    
    V = sp.zeros((N,1))
    
    z = 0
    r = 15
    delta =1
    
    while (delta > 10**-9):
        z += 1
        #print(z)
        
        #Update Policy Function    
        arg = util_grid + beta*sp.transpose(V)
        arg[index] = -10**10
        psi_ind = sp.argmax(arg,axis = 1)
        
      
        V_prev = V
        #Iterate on Value Function
        for j in sp.arange(0,r):
            V = u(W-W[psi_ind]) + beta*V[psi_ind]
    
    
        delta = sp.dot(sp.transpose(V_prev - V),V_prev-V)

    return W[psi_ind]

import solutions as sol


prob2=Problem2Real()
prob3=Problem3Real()

x=sol.Problem2()

if(np.allclose(prob2,x)):
    print("Problem2 Passed")
else:
    print("Problem2 Falied")

x=sol.Problem3()

if(np.allclose(prob3,x)):
    print("Problem3 Passed")
else:
    print("Problem3 Falied")

sol.Problem4()

import scipy as sp
from discretelognorm import discretelognorm
from matplotlib import pyplot as plt

def Problem5Real():
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
    gamma = 0.1
    beta = 0.9
    
    tol = 10**-9
    delta1 = 1+tol
    delta2 = 1+tol
    it = 0
    while ((delta1 >= tol) or (delta2 >= tol)):
        it += 1
        VE = VEprime.copy()
        VU = VUprime.copy()
        EVU = EVUprime.copy()
        
        VEprime = util_vec + beta*((1-gamma)*VE + gamma*EVU)
        arg1 = sp.repeat(sp.transpose(VE),N,0)
        arg2 = sp.repeat(EVU,N,1)
        arg = sp.array([arg2,arg1])
        VUprime = alpha_util_grid + beta*sp.amax(arg,axis = 0)
        psi = sp.argmax(arg,axis = 0)
        EVUprime = sp.dot(VUprime,f)
    
        delta1 = sp.linalg.norm(VEprime - VE)
        delta2 = sp.linalg.norm(VUprime - VU)
        #print(delta1)
        
    wr_ind = sp.argmax(sp.diff(psi), axis = 1)
    wr = w[wr_ind]
    return wr

def Problem6Real():
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
        #print(delta)    
        
    wr_ind = sp.argmax(sp.diff(psiprime), axis = 1)
    wr = w[wr_ind]
    plt.plot(w,wr)
    plt.show()
    return wr


import solutions as sol

prob1=Problem5Real()
prob2=Problem6Real()


x=sol.Problem5()

if(np.allclose(prob1,x)):
    print("Problem5 Passed")
else:
    print("Problem5 Falied")


x=sol.Problem6()

if(np.allclose(prob2,x)):
    print("Problem6 Passed")
else:
    print("Problem6 Falied")

sol.Problem7()