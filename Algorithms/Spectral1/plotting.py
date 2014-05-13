# import matplotlib
# matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
# from scipy.integrate import ode
from __future__ import division
import numpy as np
from numpy.linalg import solve
from solution import cheb
import matplotlib.pyplot as plt
import math

def deriv_matrix_exercise1():
	
	def f(x): return np.exp(x)*np.cos(6.*x)
	
	def fp(x): return np.exp(x)*(np.cos(6.*x) - 6.*np.sin(6.*x))
	
	N = 10
	D,x = cheb(N)
	f_exact, fp_exact = f(x), fp(x)
	xx = np.linspace(-1,1,200)
	plt.plot(xx,fp(xx),'-k') # Exact Derivative
	
	uu = (np.poly1d(np.polyfit(x, D.dot(f(x) ), N)))(xx)
	
	plt.plot(x,D.dot(f(x)),'*r')	# Approximation to derivative at grid points
	plt.plot(xx,uu,'-r')			# Approximation to derivative at other values
	plt.show()
	plt.clf()
	print "max error = \n", np.max(np.abs( D.dot(f(x)) - fp(x)   ))
	return 


def exercise2():
	# Solves the Poisson boundary value problem u''(x) = f(x), u(-1) = u(1) = 0
	def f(x): return np.exp(2.*x) #np.exp(4.*x)
	
	def anal_sol(x): 
		# return (np.exp(4.*x) - np.sinh(4.)*x - np.cosh(4.))/16.
		return (np.exp(2.*x) - np.sinh(2.)*x - np.cosh(2.))/4.
	
	N = 3
	D,x = cheb(N)
	
	A = np.dot(D,D)[1:N:1,1:N:1]
	u = np.zeros(x.shape); u[1:N] = solve(A,f(x[1:N:1]))
	
	xx = np.linspace(-1,1,50)
	uu = (np.poly1d(np.polyfit(x,u,N)))(xx)
	print "Max error is ", np.max(np.abs(uu - anal_sol(xx)))
	
	plt.plot(x,u,'*r'); plt.plot(xx,uu,'-r')
	plt.plot(xx,anal_sol(xx),'-k')
	plt.axis([-1.,1.,-2.5,.5]); plt.show()
	plt.clf()
	return 


def exercise3():
	# Solves the Poisson boundary value problem 
	# u''(x) + u'(x) = f(x), 
	# u(-1) = a
	# u(1) = b
	a, b = 2, -1
	def f(x): return -(b-a)/2.*np.ones(x.shape)+ np.exp(3.*x)
	
	def G(x): return (b-a)/2.*np.ones(x.shape) *x + (b+a)/2.*np.ones(x.shape)
	
	def anal_sol(x): 
		# out = ( np.exp(-x)/( 12.*(-1 +np.exp(1))*(1 + np.exp(1))*np.exp(3)   )  * 
		# ( -12.*a*np.exp(x+3.)+12.*a*np.exp(4) + 12.*b*np.exp(x+5)-12*b*np.exp(4) - 
		#   np.exp(4*x+3) + np.exp(4*x+5) - np.exp(x+8 )+ np.exp(x)- np.exp(1) + np.exp(7) 
		#   ) 
		# 	  )
		N, B = np.array([[1.,np.exp(1)],[1,np.exp(-1)]]), np.array([[a-np.exp(-3)/12.], [b-np.exp(3)/12.]])
		C = solve(N,B)
		return C[0] + C[1]*np.exp(-x) + np.exp(3.*x)/12. 
	
	
	N = 7
	D,x = cheb(N)
	
	A = np.dot(D,D)[1:N:1,1:N:1]
	u = np.zeros(x.shape); u[1:N] = solve(A + D[1:N:1,1:N:1],f(x[1:N:1]))
	
	xx = np.linspace(-1,1,50)
	uu = (np.poly1d(np.polyfit(x,u,N)))(xx)
	print "Max error is ", np.max(np.abs(uu+G(xx) - anal_sol(xx)))
	
	# plt.plot(x,u+G(x),'*k')
	# plt.plot(xx,uu+G(xx),'-r')
	plt.plot(xx,anal_sol(xx),'-k',linewidth=2.0)
	plt.xlabel('$x$',fontsize=18); plt.ylabel('$u$',fontsize=18)
	# plt.axis([-1.,1.,-2.5,.5]); 
	plt.savefig('nonzeroDirichlet.pdf')
	plt.show()
	return




# deriv_matrix_exercise1()
# exercise2()
exercise3()