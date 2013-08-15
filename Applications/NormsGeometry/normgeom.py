# -*- coding: utf-8 -*-
"""
Created on Sat May 12 16:37:18 2012

@author: -
"""

import scipy as sp
import matplotlib.pyplot as plt

def normalize(v, p):
    """Normalize vector v so it is of length one"""
    return sp.asarray(v)*(1./norm(v, p))


def norm(v, p):
    v = sp.absolute(v)
    if p > 500:
        #assume inf norm
        return v.max()
    else:
        return sp.sum(v**p)**(1./p)


def transnorm(p1, p2, npts=500):
    """Translate the unit ball from p1 to p2"""
    
    p1ball = genball(p1, npts=npts)
    newpts = sp.zeros_like(p1ball)
    print p1ball.shape
    
    for n in xrange(npts):
        transmat = sp.diag([norm(p1ball[:,n], p2)]*2)
        newpts[:,n] = p1ball[:,n].dot(transmat)

    return newpts        
    
def genball(p, npts=500):
    """Plot the unit ball for Lp space"""
    
    #divide unit ball into npts pieces
    tincr = 360. / npts
    bpts = sp.zeros((2, npts))
    
    r = 5.
    t = 0.
    n = 0
    while t < 360 and n < npts:
        p_pt = polar2cart(r, t)
        bpts[:, n] = normalize(p_pt, p)
        t += tincr
        n += 1
    
    return bpts

def plotnorm(p, npts=500, p2=None):
    if p2 is not None:
        #tranlate norm from p to p2
        bpts = transnorm(p, p2, npts=npts)
    else:
        bpts = genball(p, npts=npts)
        
    plt.plot(bpts[0], bpts[1], ',')
    plt.axis('equal')
    plt.axis([-1.5, 1.5, -1.5, 1.5])
    plt.axhline()
    plt.axvline()
    plt.show()


def polar2cart(r, t):
    return r * sp.math.cos(t), r * sp.math.sin(t)