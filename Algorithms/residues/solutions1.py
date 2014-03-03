import numpy as np
import scipy as sp
from scipy.integrate import quad
from scipy.misc import derivative
from sympy import mpmath as mp


#Problem 1
def compIntegrate(f,z,a,b,tol):
    rz = lambda t: sp.real(z(t))
    iz = lambda t: sp.imag(z(t))
    rdz = lambda t : derivative( rz, t)
    idz = lambda t : derivative( iz,  t)
    rint = lambda t : sp.real(f(z(t)))*rdz(t) - sp.imag(f(z(t)))*idz(t)
    iint = lambda t : sp.imag(f(z(t)))*rdz(t) + sp.real(f(z(t)))*idz(t)
    return quad(rint,a,b,epsabs=tol)[0] + 1j*quad(iint,a,b,epsabs=tol)[0]

def unitBall(theta):
    return sp.cos(theta)+1j*sp.sin(theta)
def line(t, z1,z2):
    r = sp.hypot(z2.imag-z1.imag,z2.real - z1.real)
    return (z2.real - z1.real)*t + z1.real + 1j*((z2.imag - z1.imag)*t + z1.imag)

#Problem 2

print compIntegrate(np.exp,unitBall,0,2*np.pi,10e-8)
print compIntegrate(sp.conj,unitBall,0,2*np.pi,10e-8)
line1 = lambda t : line(t,0,1+1j)
print compIntegrate(np.exp,line1,0,1,10e-8)
print compIntegrate(sp.conj,line1,0,1,10e-8)
line2 = lambda t : line(t,0,1)
line3 = lambda t : line(t,1,1+1j)
print compIntegrate(np.exp,line2,0,1,10e-8)
print compIntegrate(np.exp,line3,0,1,10e-8)
print compIntegrate(sp.conj,line2,0,1,10e-8)
print compIntegrate(sp.conj,line3,0,1,10e-8)
ball1 = lambda t: unitBall(t) + 1j
print compIntegrate(np.exp,ball1,3 * np.pi /2 ,2*np.pi,10e-8)
print compIntegrate(sp.conj,ball1,3 * np.pi /2 ,2*np.pi,10e-8)


#prob 3

def mpCauchy(f,C,a,b):
    integrand = lambda t,z0 : f(C(t))/(C(t)-z0)*mp.diff(C,t)
    return lambda z0 : 1/(2*np.pi*1j)*mp.quad(lambda t : integrand(t,z0),(a,b))   

#this one doesn't give very good results- it must be that the complexIntegration function is bad
def cauchy(f,C,a,b):
    return lambda z0 : 1/(2*np.pi*1j)*compIntegrate(lambda z: f(z)/(z-z0),C,a,b,10e-10) 

z = 10

C = lambda t: unitBall(t) + z

func = mpCauchy(mp.exp,C,0,2*np.pi)

print func(z)
print np.exp(z)



#prob 4
#also gives horrible answers- it's probably not correct
def laurent(f,z0,na,nb):
    C = lambda t: unitBall/10. + z0;
    N = range(na,nb+1)
    xn = [None]*(len(N))
    for i in range(0,len(xn)):
        integrand = lambda t: f(C(t))/((C(t)-z0)**(N[i]+1))*mp.diff(C,t)
        an = 1/(2*np.pi*1j)*mp.quad(integrand,(0,2*np.pi))
        xn[i] = lambda z : an*(z-z0)**N[i]
    return lambda z : sum( [x(z) for x in xn])


func = laurent(mp.exp,0,-5,5,unitBall,0,2*np.pi)
print np.exp(z)
print func(z)



#prob 8

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

cmap = plt.cm.coolwarm
cmap_r = plt.cm.coolwarm_r

def riemmann5():
    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = sp.meshgrid(X, Y)
    Z = sp.real((X+1j*Y)**(1/4.))
    ax.plot_surface(X, Y, Z, cstride=1, rstride=1, linewidth=0, cmap=cmap)
    ax.plot_surface(X, Y, -Z, cstride=1, rstride=1, linewidth=0, cmap=cmap)
    plt.show()
