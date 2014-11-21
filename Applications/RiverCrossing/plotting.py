from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import root
from scipy.interpolate import BarycentricInterpolator

from solutions import cheb


z1 = .7

def r(x):
	return -.5*(x-1.)*(x+1.)

def a(x):
	return (1.-r(x)**2.)**(-1./2)

def y(x,eps):
	return z1/2.*(x-(-1.)) + eps *(x-1.)*(x+1)*x

def yp(x,eps):
	return z1/2. + eps*(2.*x*x + (x-1.)*(x+1))


def L(x,eps):
	a_, yp_ = a(x), yp(x,eps)
	return a_*(1 + a_**2.*yp_**2.)**(1./2) - a_**2.*r(x)*yp_



def rivercurrent():
	# plt.quiver( 0, 0, 1,  0, pivot='tail', color='b', scale_units='xy',scale=1.)
	x0 = np.linspace(-1,1,8)
	r0_ = r(x0)
	x1 = np.linspace(-1,1,100)
	r_, y_ = r(x1), y(x1,-.2)
	plt.quiver( x0, np.zeros(x0.shape), np.zeros(x0.shape),  r0_, pivot='tail', color='b', scale_units='xy',scale=1.)
	plt.axis([-1.1,1.1,-.1,1.1])
	plt.savefig('rivercurrent.pdf')
	plt.plot(x1,r_,'-k')
	plt.plot(x1,y_,'-g',linewidth=2.)
	plt.plot(1,z1,'*k')
	# plt.show()
	
	N = 300
	D, x = cheb(N)
	eps = 0.
	# print "len(x) = ", x.shape
	def L_yp(x,yp_):
		a_ = a(x)
		return a_**3.*yp_*(1 + a_**2.*yp_**2.)**(-1./2) - a_**2.*r(x)
	
	def func(z):
		out = D.dot(L_yp( x,D.dot(z) ))
		out[0], out[-1] = z[0]  -z1, z[-1] - 0
		return out
	
	sol =  root(func,y(x,eps))
	print sol.success
	z = sol.x
	poly = BarycentricInterpolator(x,D.dot(z))
	num_func = lambda inpt: poly.__call__(inpt)
	
	
	def newL(x):
		a_, yp_ = a(x), num_func(x)
		return a_*(1 + a_**2.*yp_**2.)**(1./2) - a_**2.*r(x)*yp_
	
	print "The true solution is approximately "
	print integrate.quad(newL,-1,1,epsabs=1e-10,epsrel=1e-10)
	plt.plot(x,z,'-k',linewidth=2.)
	plt.show()
	
	list = range(-50,60)
	eps = np.array([item/100. for item in list])
	out = 10.*np.ones(eps.shape)
	for j in range(len(list)):
		out[j] = integrate.quad(lambda x:L(x,eps[j]),-1,1)[0]
	print "An approximate solution for the shortest time is ", np.min(out)
	print np.argmin(out), eps[np.argmin(out)]
	return 



def rivercrossing():
	x0 = np.linspace(-1,1,200)
	
	def r(x):
		return -.75*np.sqrt(1.-x**2.)
	
	
	y0 = r(x0)
	x1 = .5*(x0 + 1)
	
	def rp(x):
		return -9./16.*x/r(x)
	
	def a(x):
		return np.sqrt(1.-r(x)**2.)**(-.5)
	
	def ap():
		return r(x)*rp(x)*alpha(x)**3.
	
	
	def func(x):
		a_ = a(x)
		ap_ = ap(x)
		r_ = r(x)
		rp_ = rp(x)
		yp = D.dot(y)
		ypp = M.dot(y) # M = D.dot(D)
		out = ( (3.*ap_*yp + a_*ypp)*(1. + a_**2.*yp**2.) - 
				(a_**2.*yp)*(ap_*yp**2. + a_**2.*yp*ypp) - 
				(2./a_*ap_*r_ + rp_)*(1. + a_**2.*yp**2.)**(3./2)
		)
		return out
	
	
	plt.plot(x1,y0,'-k')
	plt.axis([-.1,1.1,-.8,.05])
	plt.show()
	



if __name__=="__main__":
	# rivercrossing()
	rivercurrent()













