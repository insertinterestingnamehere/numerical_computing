#Solutions to Policy Function Iteration Lab
import scipy as sp
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot as plt

#Problem 2
#beta = 0.99
#N = 1000
#u = lambda c: sp.sqrt(c)
#psi_ind = sp.arange(0,N)
#
#W = sp.linspace(0,1,N)
#X, Y = sp.meshgrid(W,W)
#Wdiff = Y-X
#index = Wdiff <0
#Wdiff[index] = 0
#util_grid = u(Wdiff)
#
#I = sp.sparse.identity(N)
#delta = 1
#z = 0
#while (delta > 10**-9):
#    z = z+1
#    print(z)
#    psi_prev = psi_ind.copy()
#    
#    rows = sp.arange(0,N)
#    columns = psi_ind
#    data = sp.ones(N)
#    Q = sp.sparse.coo_matrix((data,(rows,columns)),shape = (N,N))
#    Q = Q.tocsr()
#   
#   #Solve for Value Function
#    V = spsolve(I-beta*Q,u(W-W[psi_ind]))
#
#    #Find Policy Function    
#    arg = util_grid + beta*V
#    arg[index] = -10**10
#    psi_ind = sp.argmax(arg,axis = 1)
#    delta = sp.amax(sp.absolute(W[psi_ind]-W[psi_prev]))
#
#
#plt.plot(W,W[psi_ind],'r')
#plt.show()

#Problem 3
beta = 0.99
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
    print(z)
    
    #Update Policy Function    
    arg = util_grid + beta*sp.transpose(V)
    arg[index] = -10**10
    psi_ind = sp.argmax(arg,axis = 1)
    
  
    V_prev = V
    #Iterate on Value Function
    for j in sp.arange(0,r):
        V = u(W-W[psi_ind]) + beta*V[psi_ind]


    delta = sp.dot(sp.transpose(V_prev - V),V_prev-V)

plt.plot(W,W[psi_ind],'r')
plt.show()
