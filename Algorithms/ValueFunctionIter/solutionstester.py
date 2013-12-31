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