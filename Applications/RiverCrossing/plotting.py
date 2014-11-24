from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import root
from scipy.interpolate import BarycentricInterpolator

from solutions import cheb

# Solving the river crossing problem with end states
# z(-1) = 0, z(1) = z1
z1 = 5.

# Function that describes the current of the river
def r(x):
	return -.7*(x-1.)*(x+1.)

# a = (1-r)^{1/2}
def a(x):
	return (1.-r(x)**2.)**(-1./2)

# Family of possible trajectories across the river satisfying the 
# given end-states. Also their derivatives.
def y(x,eps):
	return z1/2.*(x-(-1.)) + eps *(x-1.)*(x+1)*x

def yp(x,eps):
	return z1/2. + eps*(2.*x*x + (x-1.)*(x+1))



def r_plot():
	x0 = np.linspace(-1,1,8)
	r0_ = r(x0)
	x1 = np.linspace(-1,1,100)
	r_, y_ = r(x1), y(x1,-.7)
	plt.quiver( x0, np.zeros(x0.shape), np.zeros(x0.shape),  r0_, pivot='tail', color='b', scale_units='xy',scale=1.)
	plt.axis([-1.1,1.1,0. -.05,z1 +.05])
	
	
	# # Plots the function r to ensure that the 
	# # length of the arrows above are correct.
	# plt.plot(x1,r_,'-k')
	
	# # Plots a trajectory y_ 
	y2 = np.linspace(0-.05,z1+.05,200)
	plt.plot(-np.ones(y2.shape),y2,'-k',linewidth=2.5)
	plt.plot(np.ones(y2.shape),y2,'-k',linewidth=2.5)
	plt.plot(x1,y_,'-g',linewidth=2.)
	plt.plot(-1,0,'*g',markersize=10.)
	plt.plot(1,z1,'*g',markersize=10.)
	plt.savefig('rivercurrent.pdf')
	plt.show()


def heuristic_trajectory():
	# The integrand of the functional that gives the time required to cross
	# the river on a given trajectory.
	def L(x,eps):
		a_, yp_ = a(x), yp(x,eps)
		return a_*(1 + a_**2.*yp_**2.)**(1./2) - a_**2.*r(x)*yp_
	
	
	list = range(-50,60)
	eps = np.array([item/100. for item in list])
	out = 10.*np.ones(eps.shape)
	for j in range(len(list)):
		out[j] = integrate.quad(lambda x:L(x,eps[j]),-1,1)[0]
	min_time = np.min(out)
	index = np.argmin(out)
	eps = eps[index]
	return eps, min_time  


def rivercurrent():
	N = 200  
	D, x = cheb(N)  
	eps = 0.  
	
	def L_yp(x,yp_):
		a_ = a(x)
		return a_**3.*yp_*(1 + a_**2.*yp_**2.)**(-1./2) - a_**2.*r(x)
	
	def g(z):
		out = D.dot(L_yp( x,D.dot(z) ))
		out[0], out[-1] = z[0] - z1, z[-1] - 0
		return out
	
	
	# Use the straight line trajectory as an initial guess.
	# Another option would be to use the shortest-time trajectory found
	# in heuristic_tractory() as an initial guess.
	eps, time = heuristic_trajectory()
	sol = root(g,y(x,eps))
	# sol =  root(g,z1/2.*(x-(-1.))) 
	print sol.success
	z = sol.x
	poly = BarycentricInterpolator(x,D.dot(z))
	num_func = lambda inpt: poly.__call__(inpt)
	
	
	def L(x):
		a_, yp_ = a(x), num_func(x)
		return a_*(1 + a_**2.*yp_**2.)**(1./2) - a_**2.*r(x)*yp_
	
	print "The shortest time is approximately "
	print integrate.quad(L,-1,1,epsabs=1e-10,epsrel=1e-10)[0], time
	
	x0 = np.linspace(-1,1,200)
	y0 = y(x0,eps)
	plt.plot(x0,y0,'-g',linewidth=2.)
	plt.plot(x,z,'-k',linewidth=2.)	
	plt.axis([-1,1,0.-.05,z1 + .05])
	plt.show()
	return 

if __name__=="__main__":
	r_plot()	
	# heuristic_trajectory()
	# rivercurrent()
	













