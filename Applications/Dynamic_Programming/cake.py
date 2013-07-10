import scipy as sp
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats as st
from scipy import linalg as la

def discretenorm(num, mu=0, sigma=1):
    w=sp.linspace(mu-3*sigma,mu+3*sigma,num)
    v=(w[1]-w[0])/2.
    probs=sp.zeros((num))
    for i in range(num):
        probs[i]=st.norm.cdf(w[i]+v,mu,sigma) - st.norm.cdf(w[i]-v,mu,sigma)
    return w,probs

x,y=discretenorm(7)
plt.plot(x,y)
plt.show()

W=sp.linspace(0,1,100)
Wbig=sp.zeros((100,100))
i=0
k=7
u=sp.zeros((100,100,7))
for x in W:
    Wbig[:,i]=W-x
    i=i+1
u1=(Wbig<0)*-Wbig+Wbig
u1=sp.sqrt(u1)
u1=(Wbig<0)*-10**10+u1
x,y=discretenorm(k,4*sp.sqrt(.25),.25)
for j in range(k):
    u[:,:,j]=x[j]*u1

b=.9
states=sp.zeros((1,100,7))
policy=sp.zeros((1,100,7))
i=0
d=1
while d>10**-9:
    E=states[i,:,:]*y
    V=sp.vstack(E.sum(1))*b
    Value1=V.copy()
    for x in range(100-1):
        Value1=sp.concatenate((Value1,V),1)
    Value=sp.zeros((100,100,7))
    for x in range(7):
        Value[:,:,x]=Value1.T
    total=u+Value
    temp= total.max(1)
    temp.resize(1,100,7)
    temp1= total.argmax(1)
    temp1.resize(1,100,7)
    states=sp.concatenate((states,temp),0)
    policy=sp.concatenate((policy,temp1),0)
    i=i+1
    d=la.norm(states[i-1,:,:]-states[i,:,:])

tem=states[-1,:,:]
plt.plot(W,tem)
plt.show()

tem=policy[-1,:,:]
plt.plot(W,tem/99.0)
plt.show()

plt.plot(states[:,:,1])
plt.show()

import scipy.stats as st
import scipy as sp


def tauchenhussey(N,mu,rho,sigma, baseSigma):
	
	Z     = sp.zeros((N,1))
	Zprob = sp.zeros((N,N))
	[Z,w] = gaussnorm(N,mu,baseSigma**2)
	for i in range(N):
		for j in range(N):
			EZprime    = (1-rho)*mu + rho*Z[i]
			Zprob[i,j] = w[j] * st.norm.pdf(Z[j],EZprime,sigma) / st.norm.pdf(Z[j],mu,baseSigma)
		
	for i in range(N):
		Zprob[i,:] = Zprob[i,:] / sum(Zprob[i,:])
		
	return Z.T,Zprob


def gaussnorm(n,mu,s2):
	""" 
	Find Gaussian nodes and weights for the normal distribution
	n  = # nodes
	mu = mean
	s2 = variance
	"""
	[x0,w0] = gausshermite(n)
	x = x0*sp.sqrt(2.*s2) + mu
	w = w0/sp.sqrt(sp.pi)
	return [x,w]

	
def gausshermite(n):
	"""
	Gauss Hermite nodes and weights following 'Numerical Recipes for C' 
	"""

	MAXIT = 10
	EPS   = 3e-14
	PIM4  = 0.7511255444649425

	x = sp.zeros((n,1))
	w = sp.zeros((n,1))

	m = int((n+1)/2)
	for i in range(m):
		if i == 0:
			z = sp.sqrt((2.*n+1)-1.85575*(2.*n+1)**(-0.16667))
		elif i == 1:
			z = z - 1.14*(n**0.426)/z
		elif i == 2:
			z = 1.86*z - 0.86*x[0]
		elif i == 3:
			z = 1.91*z - 0.91*x[1]
		else:
			z = 2*z - x[i-1]
		
		for iter in range(MAXIT):
			p1 = PIM4
			p2 = 0.
			for j in range(n):
				p3 = p2
				p2 = p1
				p1 = z*sp.sqrt(2./(j+1))*p2 - sp.sqrt(float(j)/(j+1))*p3
			pp = sp.sqrt(2.*n)*p2
			z1 = z
			z = z1 - p1/pp
			if sp.absolute(z-z1) <= EPS:
				break
		
		if iter>MAXIT:
			error('too many iterations'), end
		x[i,0]     = z
		x[n-i-1,0] = -z
		w[i,0]     = 2./pp/pp
		w[n-i-1,0] = w[i]
	
	x = x[::-1]
	return [x,w]

N=7
rho=.5
sigma=.5
mu=4*sigma
baseSigma=(.5+rho/4.)*sigma+(.5-rho/4.)*(sigma/sp.sqrt(1-rho**2))
x,y=tauchenhussey(N,mu,rho,sigma, baseSigma)

u=sp.zeros((100,100,7))
for j in range(k):
    u[:,:,j]=x[0,j]*u1

b=.9
statesAR=sp.zeros((1,100,7))
policyAR=sp.zeros((1,100,7))
i=0
d=1
while d>10**-9:
    E=sp.dot(statesAR[i,:,:],y.T)
    V=E*b
    Value=sp.zeros((100,100,7))
    for j in range(100):
        Value[j,:,:]=V
    total=u+Value
    temp= total.max(1)
    temp.resize(1,100,7)
    temp1= total.argmax(1)
    temp1.resize(1,100,7)
    statesAR=sp.concatenate((statesAR,temp),0)
    policyAR=sp.concatenate((policyAR,temp1),0)
    i=i+1
    d=la.norm(statesAR[i-1,:,:]-statesAR[i,:,:])


tem=statesAR[-1,:,:]
plt.plot(W,tem)
plt.show()


tem=policyAR[-1,:,:]
plt.plot(W,tem/99.0)
plt.show()

tem=policy[-1,:,:]
plt.plot(W,tem/99.0)
plt.show()