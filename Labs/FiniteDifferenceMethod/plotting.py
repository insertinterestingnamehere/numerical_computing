from __future__ import division
# import matplotlib
# matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
import matplotlib.colors as mcolors

import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
from solution import fd_order2_ode, approx_order
from numpy import cos, sin
from math import pi
from scikits import bvp_solver


def prob1():
	def u(x):
		arg = (x + pi)**2. -1.
		return sin(arg)
	
	def up(x):
		arg = (x + pi)**2. -1.
		return 2.*(x + pi)*cos(arg)
	
	
	def upp(x):
		arg = (x + pi)**2. -1.
		return 2.*cos(arg) - (2*(x + pi))**2.*sin(arg)
	
	I = [0.,1.]
	N = 5
	h = (I[1]-I[0])/N
	x = np.linspace(I[0],I[1],N+1)
	
	D,diags = np.ones((1,N-1)), np.array([0,-1,1])
	data = np.concatenate((-2.*D,1.*D,1.*D),axis=0) # This stacks up rows
	M2 = h**(-2.)*spdiags(data,diags,N-1,N-1).asformat('csr')
	ans2 = M2.dot(u(x[1:-1]))
	ans2[0] += u(I[0])*h**(-2.) # - (2.*h)**(-1.) )
	ans2[-1] += u(I[1])*h**(-2.) #+ (2.*h)**(-1.) )
	
	D,diags = np.ones((1,N-1)), np.array([-1,1])
	data = np.concatenate((-1.*D,1.*D),axis=0) # This stacks up rows
	M1 = (2.*h)**(-1.)*spdiags(data,diags,N-1,N-1).asformat('csr')
	ans1 = M1.dot(u(x[1:-1]))
	ans1[0] += -u(I[0])*(2.*h)**(-1.)
	ans1[-1] += u(I[1])*(2.*h)**(-1.)
	
	soln = (.5*upp(x) - up(x))[1:-1]
	approx = (.5*ans2 - ans1)
	
	print np.max(np.abs(soln-approx))
	
	plt.plot(x[1:-1],soln,'-k',linewidth=1.5)
	plt.plot(x[1:-1],approx,'*r',markersize=5.)
	plt.show()
	return 


def prob2():
	def bvp(epsilon, subintervals):
		# for figure2.pdf
		X,Y = fd_order2_ode(func=lambda x:-1.,a1=lambda x:epsilon,
										a2=lambda x:-1.,a3=lambda x:0.,
										a=0.,b=1., alpha=1.,beta=3.,N=subintervals)
		
		return X,Y
	
	def AnalyticSolution(x,alpha, beta,epsilon):
		  out = alpha+x+(beta-alpha-1.)*(np.exp(x/epsilon) -1.)/(np.exp(1./epsilon) -1.)   
		  return out
	
	eps, subintervals = 0.1, 20
	X,Y = bvp(eps, subintervals)
	plt.plot(X,Y,'-k',mfc="None",linewidth=2.0)
	plt.ylabel('$y$',fontsize=16)
	plt.xlabel('$x$',fontsize=16)
	# plt.axis([-.1,1.1,1-.1,3+.1])
	# plt.savefig('figure2.pdf')
	plt.show()
	plt.clf()
	num_approx = 6 # Number of Approximations
	N = 2560*np.array([2**j for j in range(num_approx)])
	approx_order(num_approx,N,bvp,eps)
	return 


def prob3():
	def bvp(epsilon, subintervals):
		# for figure3.pdf
		X,Y = fd_order2_ode(func=lambda x:np.cos(x),a1=lambda x:epsilon,
										  a2=lambda x: 0.,a3=lambda x:-4.*(np.pi-x**2.),
										  a=0.,b=np.pi/2., alpha=0.,beta=1.,N=subintervals)
		return X,Y
	
	eps, subintervals = 0.1, 400
	X,Y = bvp(eps, subintervals)
	plt.plot(X,Y,'-k',mfc="None",linewidth=2.0)
	plt.ylabel('$y$',fontsize=16)
	plt.xlabel('$x$',fontsize=16)
	# plt.axis([-.1,np.pi/2.+.1,-.1,1.5])
	# plt.savefig('figure3.pdf')
	plt.show()
	plt.clf()
	num_approx = 6 # Number of Approximations
	N = 2560*np.array([2**j for j in range(num_approx)])
	approx_order(num_approx,N,bvp,eps)
	return 


def prob4():
	def bvp(epsilon, subintervals):
		def g(x):
			out = -epsilon*pi**2.*cos(pi*x) - pi*x*sin(pi*x)
			return out
		
		X,Y = fd_order2_ode(func=g,a1=lambda x:epsilon,
										  a2=lambda x: x,a3=lambda x:0.,
										  a=-1.,b=1., alpha=-2.,beta=0.,N=subintervals)
		return X,Y
	
	eps, subintervals = 0.1, 400
	X,Y = bvp(eps, subintervals)
	plt.plot(X,Y,'-k',mfc="None",linewidth=2.0)
	
	eps, subintervals = 0.01, 400
	X,Y = bvp(eps, subintervals)
	plt.plot(X,Y,'-k',mfc="None",linewidth=2.0)
	
	eps, subintervals = 0.001, 400
	X,Y = bvp(eps, subintervals)
	plt.plot(X,Y,'-k',mfc="None",linewidth=2.0)
	
	plt.ylabel('$y$',fontsize=16)
	plt.xlabel('$x$',fontsize=16)
	plt.show()
	plt.clf()
	num_approx = 6 # Number of Approximations
	N = 2560*np.array([2**j for j in range(num_approx)])
	approx_order(num_approx,N,bvp,eps)
	return 


def prob5():
	def bvp(epsilon, subintervals):
		# X,Y = fd_order2_ode(func=lambda x: 0.,a1=lambda x:1.,
		# 								  a2=lambda x: 4.*x/(epsilon+x**2.),a3=lambda x:2./(epsilon+x**2.),
		# 								  a=-1.,b=1., alpha=1./(1.+epsilon),
		# 								  beta=1./(1.+epsilon),N=subintervals)
		
		X,Y = fd_order2_ode(func=lambda x: 0.,a1=lambda x:(epsilon+x**2.),
										  a2=lambda x: 4.*x,a3=lambda x:2.,
										  a=-1.,b=1., alpha=1./(1.+epsilon),
										  beta=1./(1.+epsilon),N=subintervals)
		return X,Y
	
	eps, subintervals = 0.05, 100
	X,Y = bvp(eps, subintervals)
	plt.plot(X,Y,'-k',mfc="None",linewidth=2.0)
	plt.ylabel('$y$',fontsize=16)
	plt.xlabel('$x$',fontsize=16)
	plt.show()
	plt.clf()
	num_approx = 6 # Number of Approximations
	N = 2560*np.array([2**j for j in range(num_approx)])
	approx_order(num_approx,N,bvp,eps)
	return 



def prob5_again():
	"""	
	Using scikits.bvp_solver to solve the bvp
	"""
	epsilon = .05
	lbc, rbc = 1./(1.+epsilon), 1./(1.+epsilon)
	
	def function1(x , y):
		return np.array([y[1] , -4.*x/(epsilon+x**2.)*y[1]-2./(epsilon+x**2.)*y[0] ]) 
	
	
	def boundary_conditions(ya,yb):
		return (np.array([ya[0] - lbc]), 
				np.array([yb[0] - rbc]))
	
	problem = bvp_solver.ProblemDefinition(num_ODE = 2,
	                                      num_parameters = 0,
	                                      num_left_boundary_conditions = 1,
	                                      boundary_points = (-1, 1),
	                                      function = function1,
	                                      boundary_conditions = boundary_conditions)
									
	solution = bvp_solver.solve(problem,
	                            solution_guess = (1./(1.+epsilon),
	                                              0.))
											
	A = np.linspace(-1.,1., 200)
	T = solution(A)
	plt.plot(A, T[0,:],'-k',linewidth=2.0)
	plt.show()
	plt.clf()
	return


def prob3_again():
	"""	
	Using scikits.bvp_solver to solve the bvp
	"""
	epsilon = .1
	lbc, rbc = 0., 1.
	
	def function1(x , y):
		return np.array([y[1] , (4./epsilon)*(pi-x**2.)*y[0] + 1./epsilon*cos(x) ]) 
	
	
	def boundary_conditions(ya,yb):
		return (np.array([ya[0] - lbc]), 
				np.array([yb[0] - rbc]))
	
	problem = bvp_solver.ProblemDefinition(num_ODE = 2,
	                                      num_parameters = 0,
	                                      num_left_boundary_conditions = 1,
	                                      boundary_points = (0., pi/2.),
	                                      function = function1,
	                                      boundary_conditions = boundary_conditions)
									
	solution = bvp_solver.solve(problem,
	                            solution_guess = (1.,
	                                              0.))
											
	A = np.linspace(0.,pi/2., 200)
	T = solution(A)
	plt.plot(A, T[0,:],'-k',linewidth=2.0)
	plt.show()
	plt.clf()
	return




# prob1()
# prob2()
# prob3()
# prob4()      # Profile correct; taken from 32 test problems for analysis
# prob5()

# Check that fd_order2_ode works for problems 3 and 5
# prob5_again()
# prob3_again()

