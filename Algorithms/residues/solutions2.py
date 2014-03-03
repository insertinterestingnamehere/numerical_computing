import numpy as np
import scipy as sp
import scipy.integrate
from sympy import mpmath as mp
import sympy as sy


#problem 1 - incomplete
def poleOrder(f,z,z0):
    return  - sy.residue(sy.diff(f,z)/f,z,z0)
def res(f,z,z0):
    n = poleOrder(f,z,z0)
    x = sy.symbols('x')
    return 1/sy.factorial(n-1) * sy.limit( 
          sy.diff( (x - z0)**n * f.subs(z,x),x,n-1 ),x,z0) 

def poleOrder(f,z,z0):
    return  - sy.residue(sy.diff(f,z)/f,z,z0)
def res(f,z,z0):
    n = poleOrder(f,z,z0)
    x = sy.symbols('x')
    return 1/sy.factorial(n-1) * sy.limit( 
          sy.diff( (x - z0)**n * f.subs(z,x),x,n-1 ),x,z0) 


#problem 2

def resFrac(num,den,z0):
    return num.subs(z,z0)/sy.diff(den).subs(z,z0)

z = sy.symbols('z')

num = z**2
den = z**6 + 1 
roots = sy.solve(den)
for root in roots:
    print "Res: f({0}) = {1} ".format(root,resFrac(num,den,root).evalf())


#problem 3
def unitBall(theta):
    return mp.cos(theta)+1j*mp.sin(theta)

def cauchyPRes (num, den):
    roots = np.roots(den)
    s = 0;
    for root in roots :
            if(root.imag > 0) :
                C = lambda t: unitBall(t)*10e-3 + root
                dC = lambda t : mp.diff(C,t)
                integrand = lambda t: np.polyval(num,C(t))/np.polyval(den,C(t))*dC(t)
                s += mp.quad(integrand ,(0,2*np.pi))
    return s
            
poly1N = np.zeros(3)
poly1N[0] = 1
poly1D = np.zeros(7)
poly1D[0] = 1
poly1D[6] = 1
print cauchyPRes(poly1N,poly1D)
poly2N = np.zeros(13)
poly2N[0] = 1
poly2N[2] = -5
poly2N[4] = 3
poly2N[6] = -16
poly2N[8] = 4
poly2N[10] = -1
poly2N[12] = -4
poly2D = np.zeros(15)
poly2D[0] = 4
poly2D[8] = 6
poly2D[14] = 12
print cauchyPRes(poly2N,poly2D)

print sp.integrate.quad( lambda z : (z**2.)/(z**6.+1),-1000.,1000.)
print sp.integrate.quad( lambda z : (z**12 - 5.*z**10 + 3.*z**8 - 16. * z**6 + 4.*z**4 - z**2 - 4)/
    (4.*z**14 +6.*z**6.+12),-1000.,1000.)

#problem 4
#everythin appears to be correct, but it doesn't give the right answers
def cauchyRes (num, den):
    roots = np.roots(den)
    s = 0;
    for root in roots :
        if(root.imag > 0) :
            C = lambda t: unitBall(t)*10e-5 + root
            dC = lambda t : diffBall(t)*10e-5
            integrand = lambda t: (num(C(t))/np.polyval(den,C(t))) *dC(t)
            s += mp.quad(integrand ,(0,2*np.pi))
    return s

den1 = np.zeros(5)
den1[0] = 1
den1[4] = 1
print cauchyRes(lambda z: mp.cos(z),den1)
den2 = np.zeros(21)
den2[0] = 1
den2[20] = 1
print cauchyRes(lambda z: (mp.sin(z))**2,den1)

#the correct answers
print sp.integrate.quad( lambda z : mp.cos(z)/np.polyval(den1,z),-1000.,1000.)
print sp.integrate.quad( lambda z : (mp.sin(z))**2/(z**20 + 1),-500.,500.)

#with sympy - it just doesn't work
z = sy.symbols('z')
f = sy.cos(z)/(z**4 +1 )
roots = sy.solve( z**4 + 1 )
s = 0
for root in roots:
    if( sy.im(root) > 0 ):
        s += (2j*sy.pi*sy.residue(f,z,root)).evalf()
print s

#problem 6
def cauchyRRRes (num, den):#Cauchy Real Roots
    roots = np.roots(den)
    s = 0;
    for root in roots : #assuming every root is real
            C = lambda t: unitBall(t)*10e-3 + root
            dC = lambda t : mp.diff(C,t)

            integrand = lambda t: 
num(C(t))/np.polyval(den,C(t))*dC(t)
            s += 1/2.*mp.quad(integrand ,(0,2*np.pi))
    return s

num = lambda z: mp.sin(z)
poly1D = np.zeros(2)
poly1D[0] =1
print cauchyRRRes(num,poly1D)
#sp.integrate.quad( lambda z : num(z)/np.polyval(poly1D,z),.0000001,100000) #doesn't converge




