#================================================
#Solutions To Value Function Iteration Lab
#================================================
#"Problem 1"
#import scipy as sp
#from matplotlib import pyplot as plt
#from matplotlib import cm
#from mpl_toolkits . mplot3d import Axes3D
#
#
#beta = 0.9;
#T = 10;
#N = 100;
#u = lambda c: sp.sqrt(c);
#W = sp.linspace(0,1,N);
#X, Y = sp.meshgrid(W,W);
#Wdiff = Y-X
#index = Wdiff <0;
#Wdiff[index] = 0;
#util_grid = u(Wdiff);
#util_grid[index] = -10**10;
#V = sp.zeros((N,T+2));
#psi = sp.zeros((N,T+1));
#
#
#for k in xrange(T,-1,-1):
#    val = util_grid + beta*sp.tile(sp.transpose(V[:,k+1]),(N,1));
#    vt = sp.amax(val, axis = 1);
#    psi_ind = sp.argmax(val,axis = 1)
#    V[:,k]    = vt;
#    psi[:,k]    = W[psi_ind];
#
#
#
#x=sp.arange(0,N)
#y=sp.arange(0,T+2)
#X,Y=sp.meshgrid(x,y)
#fig1 = plt.figure()
#ax1= Axes3D(fig1)
#ax1.plot_surface(W[X],Y,sp.transpose(V), cmap=cm.coolwarm)
#plt.show ()
#
#fig2 = plt.figure() 
#ax2 = Axes3D(fig2)
#y = sp.arange(0,T+1)
#X,Y=sp.meshgrid(x,y)
#ax2.plot_surface(W[X],Y,sp.transpose(psi), cmap = cm.coolwarm)
#plt.show()


#================================================
"Problem 2"
#import scipy as sp
#from matplotlib import pyplot as plt
#
#beta = 0.9;
#T = 1000;
#N = 100;
#u = lambda c: sp.sqrt(c);
#W = sp.linspace(0,1,N);
#X, Y = sp.meshgrid(W,W);
#Wdiff = sp.transpose(X-Y);
#index = Wdiff <0;
#Wdiff[index] = 0;
#util_grid = u(Wdiff);
#util_grid[index] = -10**10;
#V = sp.zeros((N,T+2));
#psi = sp.zeros((N,T+1));
#
#
#for k in xrange(T,-1,-1):
#    val = util_grid + beta*sp.tile(sp.transpose(V[:,k+1]),(N,1));
#    vt = sp.amax(val, axis = 1);
#    psi_ind = sp.argmax(val,axis = 1)
#    V[:,k]    = vt;
#    psi[:,k]    = W[psi_ind];
#
#
#
#plt.plot(psi[99,:])


#================================================
#"Problem 3"
#import scipy as sp
#from matplotlib import pyplot as plt
#
#beta = 0.99
#N = 1000
#u = lambda c: sp.sqrt(c)
#W = sp.linspace(0,1,N)
#X, Y = sp.meshgrid(W,W)
#Wdiff = sp.transpose(X-Y)
#index = Wdiff <0
#Wdiff[index] = 0
#util_grid = u(Wdiff)
#util_grid[index] = -10**10
#
#Vprime = sp.zeros((N,1))
#psi = sp.zeros((N,1))
#delta = 1.0
#tol = 10**-9
#it = 0
#max_iter = 500
#
#while (delta >= tol) and (it < max_iter):
#    V = Vprime
#    it += 1;
#    print(it)
#    val = util_grid + beta*sp.transpose(V)
#    Vprime = sp.amax(val, axis = 1)
#    Vprime = Vprime.reshape((N,1))
#    psi_ind = sp.argmax(val,axis = 1)
#    psi    = W[psi_ind]
#    delta = sp.dot(sp.transpose(Vprime - V),Vprime-V)


#plt.plot(W,psi)    
#plt.show()