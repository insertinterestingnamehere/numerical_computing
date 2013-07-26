import scipy as sp
from discretelognorm import discretelognorm
from matplotlib import pyplot as plt

#Problem 1 Solution============================================================
#N = 500
#w = sp.linspace(0,100,N)
#w = w.reshape(N,1)
#u = lambda c: sp.sqrt(c)
#util_vec = u(w)
#alpha = 0.5
#alpha_util = u(alpha*w)
#alpha_util_grid = sp.repeat(alpha_util,N,1)
#
#m = 20
#v = 200
#f = discretelognorm(w,m,v)
#
#VEprime = sp.zeros((N,1))
#VUprime    = sp.zeros((N,N))
#EVUprime = sp.zeros((N,1))
#gamma = 0.1
#beta = 0.9
#
#tol = 10**-9
#delta1 = 1+tol
#delta2 = 1+tol
#it = 0
#while ((delta1 >= tol) or (delta2 >= tol)):
#    it += 1
#    VE = VEprime.copy()
#    VU = VUprime.copy()
#    EVU = EVUprime.copy()
#    
#    VEprime = util_vec + beta*((1-gamma)*VE + gamma*EVU)
#    arg1 = sp.repeat(sp.transpose(VE),N,0)
#    arg2 = sp.repeat(EVU,N,1)
#    arg = sp.array([arg2,arg1])
#    VUprime = alpha_util_grid + beta*sp.amax(arg,axis = 0)
#    psi = sp.argmax(arg,axis = 0)
#    EVUprime = sp.dot(VUprime,f)
#
#    delta1 = sp.linalg.norm(VEprime - VE)
#    delta2 = sp.linalg.norm(VUprime - VU)
#    print(delta1)
#    
#wr_ind = sp.argmax(sp.diff(psi), axis = 1)
#wr = w[wr_ind]
#plt.plot(w,wr)
#plt.show()

#Problem 2 Solution============================================================
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
    print(delta)    
    
wr_ind = sp.argmax(sp.diff(psiprime), axis = 1)
wr = w[wr_ind]
plt.plot(w,wr)
plt.show()