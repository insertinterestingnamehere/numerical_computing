import scipy as sp
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats as st
from scipy import linalg as la
from scipy.sparse.linalg import spsolve

from scipy.stats import lognorm
def discretelognorm(xpoints,m,v):
    
    mu = sp.log(m**2/float(sp.sqrt(v+m**2)))
    sigma = sp.sqrt(sp.log((v/float(m**2))+1))    
    
    xmax  = sp.amax(xpoints) 
    xmin  = sp.amin(xpoints) 
    N     = sp.size(xpoints) 
    xincr = (xmax - xmin)/float(N-1)

    binnodes = sp.arange(xmin+.5*xincr,xmax + .5*xincr, xincr)
    lnormcdf  = lognorm.cdf(binnodes,sigma,0,sp.exp(mu)) 
    discrpdf  = sp.zeros((N,1))
    for i in sp.arange(N):
        if i == 0:
            discrpdf[i] = lnormcdf[i] 
        elif (i > 0) and (i < N-1):
            discrpdf[i] = lnormcdf[i] - lnormcdf[i-1] 
        elif (i == N-1):
            discrpdf[i] = discrpdf[i-1]

    return discrpdf

N=500
w=sp.linspace(0,100,N)
m=20
v=200
F = discretelognorm(w,m,v)

gamma=.1
b=.9
alpha=.5

def discretelognorm(xpoints,m,v):
    
    mu = sp.log(m**2/float(sp.sqrt(v+m**2)))
    sigma = sp.sqrt(sp.log((v/float(m**2))+1))    
    
    xmax  = sp.amax(xpoints) 
    xmin  = sp.amin(xpoints) 
    N     = sp.size(xpoints) 
    xincr = (xmax - xmin)/float(N-1)

    binnodes = sp.arange(xmin+.5*xincr,xmax + .5*xincr, xincr)
    lnormcdf  = lognorm.cdf(binnodes,sigma,0,sp.exp(mu)) 
    discrpdf  = sp.zeros((N,1))
    for i in sp.arange(N):
        if i == 0:
            discrpdf[i] = lnormcdf[i] 
        elif (i > 0) and (i < N-1):
            discrpdf[i] = lnormcdf[i] - lnormcdf[i-1] 
        elif (i == N-1):
            discrpdf[i] = discrpdf[i-1]

    return discrpdf
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats as st
from scipy import linalg as la
from scipy.sparse.linalg import spsolve
gamma=.1
b=.9
alpha=.5
N=500
w=sp.linspace(0,100,N)
m=20
v=200
F = discretelognorm(w,m,v)
VE=sp.zeros(N)
VU=sp.zeros((N,N))
policy=sp.zeros(100)
uw=sp.sqrt(w)
uwa=sp.sqrt(w*alpha)
griduwa=np.tile(uwa,(N,1)).T
i=0
d1=1
d2=1
while (d1>10**-9 or d2>10**-9):
    EVU=sp.dot(VU,F)
    tempE=uw+b*((1-gamma)*VE+gamma*EVU.T)
    X,Y=np.meshgrid(VE.T,EVU)
    theMax=np.maximum(X,Y)
    tempU=griduwa+b*theMax
    d1=la.norm(VE-tempE)
    d2=la.norm(VU-tempU)
    VE=tempE.copy()
    VU=tempU.copy()
    i=i+1

plt.imshow((X>Y)*1)
plt.gray()
plt.show()

import scipy as sp
from scipy.stats import lognorm
def discretelognorm(xpoints,m,v):
    
    mu = sp.log(m**2/float(sp.sqrt(v+m**2)))
    sigma = sp.sqrt(sp.log((v/float(m**2))+1))    
    
    xmax  = sp.amax(xpoints) 
    xmin  = sp.amin(xpoints) 
    N     = sp.size(xpoints) 
    xincr = (xmax - xmin)/float(N-1)

    binnodes = sp.arange(xmin+.5*xincr,xmax + .5*xincr, xincr)
    lnormcdf  = lognorm.cdf(binnodes,sigma,0,sp.exp(mu)) 
    discrpdf  = sp.zeros((N,1))
    for i in sp.arange(N):
        if i == 0:
            discrpdf[i] = lnormcdf[i] 
        elif (i > 0) and (i < N-1):
            discrpdf[i] = lnormcdf[i] - lnormcdf[i-1] 
        elif (i == N-1):
            discrpdf[i] = discrpdf[i-1]

    return discrpdf
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats as st
from scipy import linalg as la
from scipy.sparse.linalg import spsolve
gamma=.1
b=.9
alpha=.5
N=500
w=sp.linspace(0,100,N)
m=20
v=200
F = discretelognorm(w,m,v)
VE=sp.zeros(N)
VU=sp.zeros((N,N))
policy=sp.zeros(100)
uw=sp.sqrt(w)
uwa=sp.sqrt(w*alpha)
griduwa=np.tile(uwa,(N,1)).T
i=0
d=1
m=15
EVU=sp.dot(VU,F)
while (d>10**-9):
    X,Y=np.meshgrid(VE.T,EVU)
    psi=(X>Y)*1
    
    uv=VU.copy()
    
    for num in xrange(m):
        EVU=sp.dot(VU,F)
        VE=uw+b*((1-gamma)*VE+gamma*EVU.T)
        X,Y=np.meshgrid(VE.T,EVU)
        arg = Y*(1-psi)+X*psi
        VU = griduwa + b*arg
    psi1=(X>Y)*1
    d=la.norm(psi1-psi)
    psi=psi1.copy()
    i=i+1


plt.imshow((X>Y)*1)
plt.gray()
plt.show()

wr_ind = sp.argmax(sp.diff(psi),1)
wr = w[wr_ind]
plt.plot(w,wr)
plt.show()