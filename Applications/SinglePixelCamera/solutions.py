import matplotlib . pyplot as plt
import numpy as np

#Problem1
def SPC2(F,C):
    l=len(F)
    A=np.empty(l)
    for i in xrange(l):
        v1=F[i,1]-F[i,0]
        v2=F[i,2]-F[i,0]
        A[i]=.5*np.abs(v1[0]*v2[1]-v1[1]*v2[0])
    b=np.empty(3)
    total=np.sum(A)
    b=np.dot(A,C)/total
    return b

#Problem2
def Prob2(omega,psi,r=3):
	c=np.array([r*np.sin(psi)*np.cos(omega),r*np.sin(psi)*np.sin(omega),r*np.cos(psi)])
	return c

#Problem3
def Transform(c):
    cnorm=np.linalg.norm(c)
    t=np.arccos(-c[2]/cnorm)
    wstar=np.array([c[1]/cnorm,-c[0]/cnorm,0])
    w=wstar/np.linalg.norm(wstar)
    what=np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])
    R=np.eye(3)+np.sin(t)*what+(1-np.cos(t))*np.linalg.matrix_power(what,2)
    P=np.zeros((4,4))
    P[:3,:3]=R.T
    P[:3,-1]=np.dot(-R.T,c)
    P[-1,-1]=1
    return P

#Problem4
def Project(P,Fs,f=.5):
	l=len(Fs)
	F=np.ones((l,4,3))
	F[:,:-1,:]=Fs
    def Pc(y,f=.5):
        return np.array([y[0]/(f*y[2]),y[1]/(f*y[2])])
    Q=np.empty((l,3,2))
    for i in xrange(l):
        Q[i]=Pc(np.dot(P,F[i,:,:])).T
    return Q

#Problem5
def Visible(F,c,r=1):
	    centers=np.mean(F[:,:,:],axis=-1)
	    e=np.sqrt(np.sum((c-centers)**2,axis=1))<np.sqrt(np.sum((c)**2)+r**2)-.2
    return e

#Problem6
def SPC(F,C,omega,psi,r=3):
    c=Prob2(omega,psi,r=3)
    P=Transform(c)
    Q=Project(P,F,f=.5)
    e=Visible(F,c,r=1)
    A=np.empty(l)
    for i in xrange(l):
        v1=Pc(np.dot(P,F[i,:,1]))-Pc(np.dot(P,F[i,:,0]))
        v2=Pc(np.dot(P,F[i,:,2]))-Pc(np.dot(P,F[i,:,0]))
        A[i]=.5*np.abs(v1[0]*v2[1]-v1[1]*v2[0])
    centers=np.mean(F[:,:-1,:],axis=-1)
    M=e*A
    b=np.empty(3)
    total=np.sum(M)
    b=np.dot(M,C)/total
    return b

