import matplotlib
from mpl_toolkits.mplot3d import axes3d
# matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import numpy as np
from scipy.optimize import root
from scipy.interpolate import BarycentricInterpolator
from numpy.linalg import solve,norm
from solution import cheb_vectorized
import matplotlib.pyplot as plt
import math

def deriv_matrix_exercise1():
	
	def f(x): 
		return np.exp(x)*np.cos(6.*x)
	
	def fp(x): 
		return np.exp(x)*(np.cos(6.*x) - 6.*np.sin(6.*x))
	
	N = 10
	D,x = cheb_vectorized(N)
	f_exact, fp_exact = f(x), fp(x)
	xx = np.linspace(-1,1,200)
	plt.plot(xx,fp(xx),'-k') # Exact Derivative
	
	uu = (np.poly1d(np.polyfit(x, D.dot(f(x) ), N)))(xx)
	
	plt.plot(x,D.dot(f(x)),'*r')	# Approximation to derivative at grid points
	plt.plot(xx,uu,'-r')			# Approximation to derivative at other values
	# plt.savefig('equally_spaced_points.pdf')
	plt.show()
	plt.clf()
	print "max error = \n", np.max(np.abs( D.dot(f(x)) - fp(x)	 ))


def exercise2():
	# Solves the Poisson boundary value problem u''(x) = f(x), u(-1) = u(1) = 0
	def f(x): 
		return np.exp(2.*x)
	
	def anal_sol(x): 
		return (np.exp(2.*x) - np.sinh(2.)*x - np.cosh(2.))/4.
	
	N = 3
	D, x = cheb_vectorized(N)
	
	A = np.dot(D, D)[1:N:1,1:N:1]
	u = np.zeros(x.shape)
	u[1:N] = solve(A, f(x[1:N:1]))
	
	xx = np.linspace(-1, 1, 50)
	uu = (np.poly1d(np.polyfit(x, u, N)))(xx)
	print "Max error is ", np.max(np.abs(uu - anal_sol(xx)))
	
	plt.plot(x, u, '*r')
	plt.plot(xx, uu, '-r')
	plt.plot(xx, anal_sol(xx), '-k')
	plt.axis([-1.,1.,-2.5,.5])
	# plt.savefig('chebyshev_points.pdf')
	plt.show()
	plt.clf()



def nonzeroDirichlet():
	# Solves the Poisson boundary value problem 
	# u''(x) + u'(x) = f(x), 
	# u(-1) = a
	# u(1) = b
	a, b = 2, -1
	
	def f(x): 
		return -(b-a)/2.*np.ones(x.shape)+ np.exp(3.*x)
	
	def G(x): 
		return (b-a)/2.*np.ones(x.shape) *x + (b+a)/2.*np.ones(x.shape)
	
	def anal_sol(x): 
		# out = ( np.exp(-x)/( 12.*(-1 +np.exp(1))*(1 + np.exp(1))*np.exp(3))  * 
		# (-12.*a*np.exp(x+3.)+12.*a*np.exp(4) + 12.*b*np.exp(x+5)-12*b*np.exp(4) - 
		#	np.exp(4*x+3) + np.exp(4*x+5) - np.exp(x+8 )+ np.exp(x)- np.exp(1) + np.exp(7) 
		#	) 
		#	  )
		N, B = np.array([[1.,np.exp(1)],[1,np.exp(-1)]]), np.array([[a-np.exp(-3)/12.], [b-np.exp(3)/12.]])
		C = solve(N,B)
		return C[0] + C[1]*np.exp(-x) + np.exp(3.*x)/12. 
	
	
	N = 7
	D, x = cheb_vectorized(N)
	
	A = np.dot(D, D)[1:N:1,1:N:1]
	u = np.zeros(x.shape)
	u[1:N] = solve(A + D[1:N:1,1:N:1], f(x[1:N:1]))
	
	xx = np.linspace(-1, 1, 50)
	uu = (np.poly1d(np.polyfit(x,u,N)))(xx)
	print "Max error is ", np.max(np.abs(uu + G(xx) - anal_sol(xx)))
	
	# plt.plot(x,u+G(x), '*k')
	# plt.plot(xx,uu+G(xx), '-r')
	plt.plot(xx, anal_sol(xx))
	plt.xlabel('$x$')
	plt.ylabel('$u$')
	# plt.axis([-1.,1.,-2.5,.5])
	# plt.savefig('nonzeroDirichlet.pdf')
	plt.show()
	plt.clf()



def nonlinear_minimal_area_surface_of_revolution():
	l_bc, r_bc = 1., 7.
	N = 80
	D, x = cheb_vectorized(N)
	M = np.dot(D, D)
	guess = 1. + (x--1.)*((r_bc - l_bc)/2.)
	N2 = 50
	
	def pseudospectral_ode(y):
		out = np.zeros(y.shape)
		yp, ypp = D.dot(y), M.dot(y)
		out = y*ypp - 1. - yp**2.
		out[0], out[-1] = y[0] - r_bc, y[-1] - l_bc
		return out
	
	u = root(pseudospectral_ode,guess,method='lm',tol=1e-9)
	num_sol = BarycentricInterpolator(x,u.x)
	
	# Up to this point we have found the numerical solution 
	# using the pseudospectral method. In the code that follows
	# we check that solution with the analytic solution, 
	# and graph the results
	
	def f(x):
		return np.array([ x[1]*np.cosh((-1.+x[0])/x[1])-l_bc, 
						 x[1]*np.cosh((1.+x[0])/x[1])-r_bc])
	
	
	parameters = root(f,np.array([1.,1.]),method='lm',tol=1e-9)
	A, B = parameters.x[0], parameters.x[1]
	def analytic_solution(x):
		out = B*np.cosh((x + A)/B)
		return out
	
	
	xx = np.linspace(-1, 1, N2)
	uu = num_sol.__call__(xx)
	# print "Max error is ", np.max(np.abs(uu - analytic_solution(xx)))
	plt.plot(x,guess,'-b')
	plt.plot(xx, uu, '-r')						# Numerical solution via 
												# the pseudospectral method
	plt.plot(xx, analytic_solution(xx), '*k')   # Analytic solution
	plt.axis([-1.,1.,l_bc-1.,r_bc +1.])
	# plt.show()
	plt.clf()
	
	theta = np.linspace(0,2*np.pi,N2)	
	X,Theta = np.meshgrid(xx,theta,indexing='ij')
	print "\nxx = \n", xx
	print "\nuu = \n", uu
	F = uu[:,np.newaxis] +np.zeros(uu.shape)
	print "\nX = \n", X
	print "\nTheta = \n", Theta
	print "\nF = \n", F
	Y = F*np.cos(Theta)
	Z = F*np.sin(Theta)
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	# X, Y, Z = axes3d.get_test_data(0.05)
	ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
	print ax.azim, ax.elev
	ax.azim=-65; ax.elev = 0
	# ax.view_init(elev=-60, azim=30)
	# plt.savefig('minimal_surface.pdf')
	plt.show()
	
	
	

if __name__ == "__main__":
	# deriv_matrix_exercise1()
	# exercise2()
	# nonzeroDirichlet()
	nonlinear_minimal_area_surface_of_revolution()