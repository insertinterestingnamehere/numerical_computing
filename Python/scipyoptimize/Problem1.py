# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from scipy import optimize as opt
import numpy as np

# <rawcell>

# my system couldn't find the dogleg and trust-ncg methods, and the Newton-CG required the Jacobian - so I didn't do it

# <codecell>

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
opt.minimize(opt.rosen, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})

# <codecell>

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
opt.minimize(opt.rosen, x0, method='Powell', options={'xtol': 1e-8, 'disp': True})

# <codecell>

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
opt.minimize(opt.rosen, x0, method='CG', options={'disp': True})

# <codecell>

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
opt.minimize(opt.rosen, x0, method='BFGS', options={'xtol': 1e-8, 'disp': True})

# <codecell>

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
opt.minimize(opt.rosen, x0, method='Newton-CG', jac=opt.rosen_der,options={'xtol': 1e-8, 'disp': True})

# <codecell>

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
opt.minimize(opt.rosen, x0, method='Anneal', options={'xtol': 1e-8, 'disp': True})

# <codecell>

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
opt.minimize(opt.rosen, x0, method='L-BFGS-B', options={'xtol': 1e-8, 'disp': True})

# <codecell>

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
opt.minimize(opt.rosen, x0, method='TNC', options={'xtol': 1e-8, 'disp': True})

# <codecell>

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
opt.minimize(opt.rosen, x0, method='COBYLA', options={'xtol': 1e-8, 'disp': True})

# <codecell>

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
opt.minimize(opt.rosen, x0, method='SLSQP', options={'xtol': 1e-8, 'disp': True})

# <codecell>

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
opt.minimize(opt.rosen, x0, method='dogleg', options={'xtol': 1e-8, 'disp': True})

# <codecell>

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
opt.minimize(opt.rosen, x0, method='trust-ncg', options={'xtol': 1e-8, 'disp': True})

# <codecell>


