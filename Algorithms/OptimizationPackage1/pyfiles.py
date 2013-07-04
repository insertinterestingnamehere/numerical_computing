from scipy import optimize as opt
import numpy as np

def rosen(x):
    #return (1-x)^2 + 100(y-x^2)^2
	return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def rast(x):
    return 10*2 + sum(x[:-1]**2.0 - 10*np.cos(2*np.pi*x[:-1])+x[:1]**2.0 - 10*np.cos(2*np.pi*x[:1]))

##x= np.array([21,15])
x = [1.3,.7,.8,1.9,1.2]
##x = [-1.2, 3, 12]
f = opt.rosen
fprime = opt.rosen_der

##print opt.fmin(f, x)
##print opt.fmin_powell(rosen,x)
##print opt.fmin_cg(rosen,x)
##print opt.fmin_bfgs(rosen,x)
##print opt.fmin_ncg(rosen,x, fprime, fhess=opt.rosen_hess)


def g(x):
    return x**2 + 10*np.sin(x)
grid = (-1,2,.01)
print opt.brute(g,(grid,))
##
##
##x = [.5]
##print opt.fmin_l_bfgs_b(g,x)
##print opt.fmin_bfgs(g,x)
