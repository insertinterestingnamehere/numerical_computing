# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
from scipy import optimize as opt
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

u = np.linspace(0, 2 * np.pi, 10)
v = np.linspace(0, np.pi, 10)

X = 5* np.outer(np.cos(u), np.sin(v))+1
Y = 5* np.outer(np.sin(u), np.sin(v))

R = np.sqrt((X+1)**2 + Y**2)
Z = R**2 *(1+ np.sin(4*R)**2)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)

plt . show ()

# <codecell>

def multimin(x):
    r = np.sqrt((x[0]+1)**2 + x[1]**2)
    return r**2 *(1+ np.sin(4*r)**2)

# <codecell>

x0 = [-2,-2]
res = opt.fmin(multimin, x0, xtol=1e-8, disp=True)
print res
print multimin(res)
print multimin([-1,0])

# <codecell>

x0 = [-2,-2]
opt.minimize(multimin, x0, method='Powell', options={'xtol': 1e-8, 'disp': True})

# <codecell>

opt.basinhopping(multimin,x0,stepsize=0.5,options={'method':'Neader-Mead','xtol': 1e-8}

