import scipy as sp
import numpy as np
import matplotlib . pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import linalg as la

W=sp.linspace(0,1,100)
Wbig=sp.zeros((100,100))
i=0
for x in W:
    Wbig[:,i]=W-x
    i=i+1
u=(Wbig<0)*-Wbig+Wbig
u=sp.sqrt(u)
u=(Wbig<0)*-10**10+u


T=1000
states=sp.zeros((100,T+2))
policy=sp.zeros((100,T+2))
for i in sp.arange(T+1,0,step=-1):
    V=sp.vstack(states[:,i])*b
    Value=V.copy()
    for x in range(100-1):
        Value=sp.concatenate((Value,V),1)
    total=u+Value.T
    states[:,i-1]=total.max(1)
    policy[:,i-1]=total.argmax(1);

plt.plot(states)
plt.show()

plt.plot(W,policy/99.0)
plt.show()

N=100
b=.9
W=sp.linspace(0,1,100)
Wbig=sp.zeros((100,100))
i=0
for x in W:
    Wbig[:,i]=W-x
    i=i+1
u=(Wbig<0)*-Wbig+Wbig
u=sp.sqrt(u)
u=(Wbig<0)*-10**10+u
states=sp.zeros(100)
policy=sp.zeros(100)
i=0
d=1
while d>10**-9:
    V=states.copy()
    Value=sp.zeros((N,N))
    for j in range(N):
        Value[j,:]=V.T*b
    total=u+Value
    temp= total.max(1)
    temp1= total.argmax(1)
    d=la.norm(states-temp)
    policy=temp1.copy()
    states=temp.copy()

plt.plot(W,policy/99.0)
plt.show()

plt.plot(states)
plt.show()