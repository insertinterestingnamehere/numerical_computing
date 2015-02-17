import numpy as np
from scipy import optimize as op
from scipy.linalg import norm, solve
from math import sqrt, sin, cos
from matplotlib import pyplot as plt

def trustRegion(f,g,B,subproblem,x,r,rhat,eta,gtol=1e-5):
    gx = g(x)
    while norm(gx) >= gtol:

        Bx = B(x)
        p = subproblem(gx,Bx,r)
        rho = (f(x)-f(x+p))/(-np.dot(gx,p)-.5*np.dot(p,np.dot(Bx,p)))
        if rho < .25:
            r = .25*r
        elif rho > .75 and np.allclose(r,norm(p)):
            r = min(2*r,rhat)
        if rho > eta:
            x += p
        gx = g(x)
    return x
def dogleg(gx,Bx,r):
    pB = solve(Bx,-gx)
    if norm(pB) <= r:
        return pB
    pU = -(np.dot(gx,gx)/np.dot(gx,np.dot(Bx,gx)))*gx
    npU = norm(pU)
    if npU > r:
        return (r/npU)*pU
    #print "second leg"
    a = np.dot(pB-pU,pB-pU)
    b = 2*np.dot(pU,pB-pU)
    c = np.dot(pU,pU) - r**2
    tau = 1 + (-b + sqrt(b**2-4*a*c))/(2*a)
    return pU + (tau-1)*(pB-pU)  
