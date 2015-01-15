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



def current_plot():
	x0 = np.linspace(-1,1,8)
	r0_ = r(x0)
	x1 = np.linspace(-1,1,100)
	r_, y_ = r(x1), z1/2.*(x1-(-1.)) #y(x1,-.7)
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
	# plt.savefig('rivercurrent.pdf')
	# plt.show()
	plt.clf()


def time_functional():
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


def trajectory():
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
	eps, time = time_functional()
	sol = root(g,y(x,eps))
	# sol =  root(g,z1/2.*(x-(-1.))) 
	# print sol.success
	z = sol.x
	poly = BarycentricInterpolator(x,D.dot(z))
	num_func = lambda inpt: poly.__call__(inpt)
	
	
	def L(x):
		a_, yp_ = a(x), num_func(x)
		return a_*(1 + a_**2.*yp_**2.)**(1./2) - a_**2.*r(x)*yp_
	
	# print "The shortest time is approximately "
	# print integrate.quad(L,-1,1,epsabs=1e-10,epsrel=1e-10)[0]
	
	# x0 = np.linspace(-1,1,200)
	# y0 = y(x0,eps)
	# plt.plot(x0,y0,'-g',linewidth=2.,label='Initial guess')
	# plt.plot(x,z,'-b',linewidth=2.,label="Numerical solution")	
	# ax = np.linspace(0-.05,z1+.05,200)
	# plt.plot(-np.ones(ax.shape),ax,'-k',linewidth=2.5)
	# plt.plot(np.ones(ax.shape),ax,'-k',linewidth=2.5)
	# plt.plot(-1,0,'*b',markersize=10.)
	# plt.plot(1,z1,'*b',markersize=10.)
	# plt.xlabel('$x$',fontsize=18)
	# plt.ylabel('$y$',fontsize=18)
	# plt.legend(loc='best')
	# plt.axis([-1.1,1.1,0. -.05,z1 +.05])
	# # plt.savefig("minimum_time_rivercrossing.pdf")
	# plt.show()
	# plt.clf()
	return x, z, N

def angle():
	x,z, N = trajectory()
	num = int(N/20)
	D, _ = cheb(N)  
	yp = D.dot(z)
	r_ = r(x)
	
	theta1 = np.arctan(yp) # This is incorrect
	theta2 = (1-r_**2.)**(-1)*(-r_*yp + np.sqrt( (r_*yp)**2. + (1-r_**2)*(1. + yp**2.)))
	theta2 = np.arccos(1./theta2)
	# plt.plot(x,theta1,'-g',linewidth=2.)
	plt.plot(x,theta2,'-b',linewidth=2.)
	plt.xlabel('$x$',fontsize=18)
	plt.ylabel(r'$\theta$',fontsize=18)
	plt.yticks([0, np.pi/6, np.pi/3.,np.pi/2.],
	           ['$0$', r'$\frac{\pi}{6}$', r'$\frac{\pi}{3}$', r'$\frac{\pi}{2}$'])
	# plt.savefig('trajectory_angle.pdf')
	plt.show()
	plt.clf()
	
	theta = theta1
	# Checking that the angle is correct.
	print 0., np.min(theta), np.max(theta), np.pi/2.
	xdist, ydist = np.cos(theta), np.sin(theta)
	plt.quiver( x[::num], z[::num], xdist[::num],  ydist[::num], 
				pivot='tail', color='b', scale_units='xy',scale=3., angles='xy')
	# plt.plot(x,z,'-k',linewidth=2.)
	plt.axis([-1.1,1.1,0. -.5,z1 +1.])
	plt.show()

if __name__=="__main__":
	# current_plot()	
	# time_functional()
	# trajectory()
	angle()













