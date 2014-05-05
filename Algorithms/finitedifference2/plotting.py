from __future__ import division
import numpy as np
from solution import general_secondorder_ode_fd, poisson_square
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm


def Exercise1():
	def problem1(epsilon=.4,subintervals=20):
	    X,Y = general_secondorder_ode_fd(func=lambda x:-1.,a1=lambda x:epsilon,
	                                    a2=lambda x: -1.,a3=lambda x:0.,
	                                    a=0.,b=1., alpha=1.,beta=3.,N=subintervals)
	    return X,Y
	
	def AnalyticSolution(x,alpha, beta,epsilon):
	    out = alpha+x+(beta-alpha-1.)*(np.exp(x/epsilon) -1.)/(np.exp(1./epsilon) -1.)   
	    return out
	
	
	eps, subintervals = 0.005, 400
	X,Y = problem1(eps, subintervals)
	plt.plot(X,Y,'-k',mfc="None",linewidth=2.0)
	plt.axis([0.,1.1,.8,3.2])
	plt.show()
	
	
	N = np.array([5,10,20,40,80,160,320,640,1280])
	h, MaxError = 2./N, np.ones(len(N))
	eps = .4
	for j in range(len(N)): 
		Mesh, Sol = problem1(epsilon=eps,subintervals=N[j])
		MaxError[j] = (max(abs( Sol- AnalyticSolution(Mesh,alpha=1.,beta=3.,epsilon=eps) ) ) )
		
	plt.loglog(h,h**(2.),'-k',label="$h^{\, 2}$" )
	plt.loglog(h,MaxError,'.-r',label="$E(h^2)$")
	plt.xlabel("$h$")
	plt.legend(loc='best')
	plt.show()
	print "Order of the Approximation is about ", ( (np.log(MaxError[0]) - 
                np.log(MaxError[-1]) )/( np.log(h[0]) - np.log(h[-1]) ) )
	return 



def ExercisePoisson():
	from numpy import sin, cos, pi
	# Domain: [0,1]x[0,1]
	a1,b1 = 0.,1.
	c1,d1 = 0.,1.
	n=100
	# Example1: Laplace's equation (Poisson with no source)
	def bcs(x,y): return x**3.
	def source(x,y): return 0.
	
	
	# # Example2: Poisson's equation
	# def bcs(x,y): return sin(pi*x)*cos(2.*pi*y)
	# def source(x,y): return -5.*(pi**2.)*bcs(x,y)
	
	# # Example3: Poisson's equation
	# def bcs(x,y): return sin(2.*pi*y)*cos(pi*x)
	# def source(x,y): return -5.*(pi**2.)*bcs(x,y)
	
	# # Example4: Poisson's equation
	# def bcs(x,y): return 1.-x +x*y + (1./2)*sin(pi*x)*sin(pi*y)
	# 
	# def source(x,y): return -(pi**2)*sin(pi*x)*sin(pi*y)
	
	
	
	z=poisson_square(a1,b1,c1,d1,n,bcs,source)
	print '---------------'
	print "Computation successful"
	print '---------------'
	# Plotting data
	fig = plt.figure()
	#---- First subplot: Numerical Solution
	# ax = fig.add_subplot(121, projection='3d')
	ax = fig.gca(projection='3d')
	ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
	x, y = np.linspace(a1,b1,n+1), np.linspace(c1,d1,n+1)
	xv, yv = np.meshgrid(x, y)	
	xv, yv = xv.T, yv.T
	surf = ax.plot_surface(xv, yv, z, rstride=2, cstride=2, cmap=cm.coolwarm,
	        linewidth=0, antialiased=False)
	# #---- Second subplot: Exact Solution
	# ax2 = fig.add_subplot(122, projection='3d')
	# ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
	# surf2 = ax2.plot_surface(xv, yv, bcs(xv,yv), rstride=2, cstride=2, cmap=cm.coolwarm,
	# 	        linewidth=0, antialiased=False)
	print "Maximum Error = \n", np.max(np.abs( z-bcs(xv,yv) ) )
	# plt.savefig('./Poisson_solution.png')
	plt.clf()
	# plt.show()
	
	
	# if True: return
	
	num_approx = 7 # Number of Approximations
	N = np.array([10*2**(j) for j in range(num_approx)])
	h, max_error = (b1-a1)/N[:-1], np.ones(num_approx-1)
	
	num_sol_best = poisson_square(a1,b1,c1,d1,N[-1],bcs,source)
	for j in range(len(N)-1): 
	    num_sol = poisson_square(a1,b1,c1,d1,N[j],bcs,source)
	    max_error[j] = np.max(np.abs( num_sol- num_sol_best[::2**(num_approx-j-1), ::2**(num_approx-j-1)] ) )
	plt.loglog(h,max_error,'.-r',label="$E(h)$")
	plt.loglog(h,h**(2.),'-k',label="$h^{\, 2}$")
	plt.xlabel("$h$")
	plt.legend(loc='best')
	print "The order of the finite difference approximation is about ", ( (np.log(max_error[0]) - 
	    np.log(max_error[-1]) )/( np.log(h[0]) - np.log(h[-1]) ) ), "."
	plt.savefig('./Poisson_Error.png')
	plt.show()
	return 






# Exercise1()

ExercisePoisson()