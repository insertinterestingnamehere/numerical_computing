import scipy as sp
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits . mplot3d import Axes3D

def Problem1Real():
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

    
    return V,psi

def Problem2Real():
    beta = 0.9;
    T = 1000;
    N = 100;
    u = lambda c: sp.sqrt(c);
    W = sp.linspace(0,1,N);
    X, Y = sp.meshgrid(W,W);
    Wdiff = sp.transpose(X-Y);
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

    return psi

def Problem3Real():
    beta = 0.9
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
    psi = sp.zeros((N,1))
    delta = 1.0
    tol = 10**-9
    it = 0
    max_iter = 500
    
    while (delta >= tol) and (it < max_iter):
        V = Vprime
        it += 1;
        #print(it)
        val = util_grid + beta*sp.transpose(V)
        Vprime = sp.amax(val, axis = 1)
        Vprime = Vprime.reshape((N,1))
        psi_ind = sp.argmax(val,axis = 1)
        psi    = W[psi_ind]
        delta = sp.dot(sp.transpose(Vprime - V),Vprime-V)
    
    return psi

import solutions as sol

prob11,prob12=Problem1Real()
prob2=Problem2Real()
prob3=Problem3Real()

x,y=sol.Problem1()

if(np.allclose(prob11,x) and np.allclose(prob12,y)):
    print("Problem1 Passed")
else:
    print("Problem1 Falied")


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

import scipy as sp
import discretenorm
from matplotlib import pyplot as plt
from mpl_toolkits . mplot3d import Axes3D
from matplotlib import cm
from scipy import stats as st
import tauchenhussey

def discretenormR(K,mu,sigma):
    upper = mu + 3*sigma
    lower = mu - 3*sigma
    
    inc = (upper - lower)/float(K)

    left = lower    
    prob = sp.zeros(K)
    eps  = sp.zeros(K)
    for k in range(K):
        prob[k] = st.norm.cdf(left+inc,mu,sigma) - st.norm.cdf(left,mu,sigma)
        eps[k] = left + .5*inc
        left = left + inc
        
    return prob, eps

def Problem5Real():
    sigma = .5
    mu = 4*sigma
    K = 7
    Gamma, eps = discretenorm(K,mu,sigma)
    
    N = 100
    W = sp.linspace(0,1,N)
    V = sp.zeros((N,K))
    
    u = lambda c: sp.sqrt(c)
    beta = 0.99
    
    X,Y= sp.meshgrid(W,W)
    Wdiff = Y-X
    index = Wdiff < 0
    Wdiff[index] = 0
    
    util_grid = u(Wdiff)
    
    util3 = sp.tile(util_grid[:,:,sp.newaxis],(1,1,K))
    eps_grid = eps[sp.newaxis,sp.newaxis,:]
    eps_util = eps_grid*util3
    
    Gamma_grid = Gamma[sp.newaxis,:]
    
    delta = 1
    Vprime = V
    z = 0
    while (delta > 10**-9):
        z= z+1
        V = Vprime
        gamV = Gamma_grid*V
        Expval = sp.sum(gamV,1)
        Exp_grid = sp.tile(Expval[sp.newaxis,:,sp.newaxis],(N,1,K))
        arg = eps_util+beta*Exp_grid
        arg[index] = -10^10
        Vprime = sp.amax(arg,1)
        psi_ind = sp.argmax(arg,1)
        psi = W[psi_ind]
        delta = sp.linalg.norm(Vprime - V)
        
    
    return Vprime,psi

def Problem6Real():
    sigma = .5
    mu = 4*sigma
    rho = .5
    sigmaZ = sigma/sp.sqrt(1-rho**2)
    w = 0.5 + rho/4
    baseSigma = w*sigma +(1-w)*sigmaZ
    K = 7
    eps, Gamma = tauchenhussey.tauchenhussey(K,mu,rho,sigma, baseSigma)
    eps = sp.reshape(eps,K)
    
    N = 100
    W = sp.linspace(0,1,N)
    V = sp.zeros((N,K))
    
    u = lambda c: sp.sqrt(c)
    beta = 0.9
    
    X,Y= sp.meshgrid(W,W)
    Wdiff = Y-X
    index = Wdiff < 0
    Wdiff[index] = 0
    
    util_grid = u(Wdiff)
    
    
    util3 = sp.tile(util_grid[:,:,sp.newaxis],(1,1,K))
    eps_grid = eps[sp.newaxis,sp.newaxis,:]
    eps_util = eps_grid*util3
    
    delta = 1
    Vprime = V
    z=0
    while (delta>10**-9):
        z=z+1
        V = Vprime
        Expval = sp.dot(V,sp.transpose(Gamma))
        Exp_grid = sp.tile(Expval[sp.newaxis,:,:],(N,1,1))
        arg = eps_util+beta*Exp_grid
        arg[index] = -10^10
        Vprime = sp.amax(arg,1)
        psi_ind = sp.argmax(arg,1)
        psi = W[psi_ind]
        delta = sp.linalg.norm(Vprime - V)
    
    return Vprime,psi

import solutions as sol

prob11,prob12=discretenormR(7,0,.5)
prob21,prob22=Problem2Real()
prob31,prob32=Problem3Real()

x,y=sol.discretenorm(7,0,.5)

if(np.allclose(prob11,x) and np.allclose(prob12,y)):
    print("Problem4 Passed")
else:
    print("Problem4 Falied")


x,y=sol.Problem5()

if(np.allclose(prob21,x) and np.allclose(prob22,y)):
    print("Problem5 Passed")
else:
    print("Problem5 Falied")

x,y=sol.Problem6()

if(np.allclose(prob31,x) and np.allclose(prob32,y)):
    print("Problem6 Passed")
else:
    print("Problem6 Falied")