# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

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

