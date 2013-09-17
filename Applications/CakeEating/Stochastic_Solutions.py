import scipy as sp
import discretenorm
from matplotlib import pyplot as plt
from mpl_toolkits . mplot3d import Axes3D
from matplotlib import cm
from scipy import stats as st
import tauchenhussey


#Problem 1=====================================================================
def discretenorm(K,mu,sigma):
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

#Problem 2=====================================================================
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
    
    
x=sp.arange(0,N)
y=sp.arange(0,K)
X,Y=sp.meshgrid(x,y)
fig1 = plt.figure()
ax1= Axes3D(fig1)
ax1.plot_surface(W[X],Y,sp.transpose(Vprime), cmap=cm.coolwarm)
plt.show ()

fig2 = plt.figure() 
ax2 = Axes3D(fig2)
y = sp.arange(0,K)
X,Y=sp.meshgrid(x,y)
ax2.plot_surface(W[X],Y,sp.transpose(psi), cmap = cm.coolwarm)
plt.show()

#Problem 3=====================================================================
sigma = .5
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
    
x=sp.arange(0,N)
y=sp.arange(0,K)
X,Y=sp.meshgrid(x,y)
fig1 = plt.figure()
ax1= Axes3D(fig1)
ax1.plot_surface(W[X],Y,sp.transpose(Vprime), cmap=cm.coolwarm)
plt.show ()

fig2 = plt.figure() 
ax2 = Axes3D(fig2)
y = sp.arange(0,K)
X,Y=sp.meshgrid(x,y)
ax2.plot_surface(W[X],Y,sp.transpose(psi), cmap = cm.coolwarm)
plt.show()