# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import scipy.optimize as opt
import numpy as np
from matplotlib import pyplot as plt

# <codecell>

#x0 = np.array([4., -2.5])
#opt.minimize(opt.rosen, x0, method='Nelder-Mead', options={'xtol': 1e-8, 'disp': True})

# <codecell>

#opt.minimize(opt.rosen, x0, method='Powell', options={'xtol': 1e-8, 'disp': True})

# <codecell>

#opt.minimize(opt.rosen, x0, method='CG', options={'disp': True})

# <codecell>

#opt.minimize(opt.rosen, x0, method='BFGS', options={ 'disp': True})

# <codecell>

#opt.minimize(opt.rosen, x0, method='Newton-CG', jac=opt.rosen_der,options={'xtol': 1e-8, 'disp': True})

# <codecell>

#opt.minimize(opt.rosen, x0, method='Newton-CG', jac=opt.rosen_der,hess=opt.rosen_hess,options={'xtol': 1e-8, 'disp': True})

# <codecell>

#opt.minimize(opt.rosen, x0, method='Anneal', options={'disp': True})

# <codecell>

#opt.minimize(opt.rosen, x0, method='L-BFGS-B', options={'xtol': 1e-8, 'disp': True})

# <codecell>

#opt.minimize(opt.rosen, x0, method='TNC', options={'xtol': 1e-8, 'disp': True})

# <codecell>

#opt.minimize(opt.rosen, x0, method='COBYLA', options={'xtol': 1e-8, 'disp': True})

# <codecell>

#opt.minimize(opt.rosen, x0, method='SLSQP', options={'xtol': 1e-8, 'disp': True})

# <codecell>

def Problem1():
    print 'The least iterations was Powell'
    print 'COBYLA and Anneal did not converge'

# <codecell>

def Problem2():
    def multimin(x):
        r = np.sqrt((x[0]+1)**2 + x[1]**2)
        return r**2 *(1+ np.sin(4*r)**2)
    x0 = [-2,-2]
    #opt.basinhopping(multimin,x0,niter=100,stepsize=0.5,minimizer_kwargs={'method':'Nelder-Mead'})
    res=opt.basinhopping(multimin,x0,niter=100,stepsize=0.2,minimizer_kwargs={'method':'Nelder-Mead'})
    return res.fun

# <codecell>

def Problem3():
    def func(x):
        return [-x[0]+x[1]+x[2],1+x[0]**3-x[1]**2+x[2]**3,-2-x[0]**2+x[1]**2+x[2]**2]
    def jac(x):
        return np.array([[-1,1,1],[3*x[0]**2,-2*x[1],3*x[2]**2],[-2*x[0],2*x[1],2*x[2]]])
    sol = opt.root(func, [0, 0, 0], jac=jac, method='hybr')
    return sol.x

# <codecell>

def Problem4():
    data = np.loadtxt("heating.txt")
    T_a = 290
    P = 59.34
    def func(t, gamma, C, A):
        return T_a + P/gamma + A*np.exp(-gamma/C*t)
    popt, pcov = opt.curve_fit(func, data[:,0],data[:,1])
    #fitFunc = lambda t: func(t,popt[0],popt[1],popt[2])
    #fitData = fitFunc(data[:,0])

    #plt.plot(data[:,0],fitData)
    #plt.scatter(data[:,0],data[:,1],marker='.',linewidths=0,color="black")
    #plt.show()
    return popt

