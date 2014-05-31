from __future__ import division
import numpy as np
from solution import general_secondorder_ode_fd, poisson_square
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve

def example():
	# First Code block in the lab manual
	# from __future__ import division
	# import numpy as np
	from scipy.sparse import spdiags
	from scipy.sparse.linalg import spsolve
	
	def bvp(func,a=0.,b=2.,alpha=-1.,beta=3.,N = 5):
		h = (b-a)/N 				# The length of each subinterval
		
		# Initialize and define the vector F on the right
		F = np.empty(N-1.)			
		F[0] = func(a+1.*h)-alpha*h**(-2.)
		F[N-2] = func(a+(N-1)*h)-beta*h**(-2.)
		for j in xrange(1,N-2): 
			F[j] = func(a + (j+1)*h)
			
		# Here we define the arrays that will go on the diagonals of A
		D0, D1 = -2.*np.ones((1,N-1)), np.ones((1,N-1))  
		# Next we concatenate the arrays, and specify on which diagonals they will be placed
		diags = np.array([0,-1,1])
		data = np.concatenate((D0,D1,D1),axis=0) 
		A=h**(-2.)*spdiags(data,diags,N-1,N-1).asformat('csr')
		
		# We create and return the numerical approximation
		U = spsolve(A,F)
		U = np.concatenate( ( np.array([alpha]), U, np.array([beta]) ) )
		return np.linspace(a,b,N+1), U
	
	x, y = bvp(lambda x:(-3.*np.sin(x)), a=0., b=2., alpha=-2., beta=1, N=30)
	
	
	# Second code block in the lab manual
	import matplotlib.pyplot as plt
	a, b = 0., 1.
	num_approx = 10 # Number of Approximations
	N = np.array([5*2**j for j in range(num_approx)])
	h, max_error = (b-a)/N[:-1], np.ones(num_approx-1)
	
	mesh_best, num_sol_best = bvp(lambda x:-3.*np.sin(x), a, b, alpha=-2., beta=1, N=N[-1])
	for j in range(len(N)-1): 
	    mesh, num_sol = bvp(lambda x:-3.*np.sin(x), a, b, alpha=-2., beta=1, N=N[j])
	    max_error[j] = np.max(np.abs( num_sol- num_sol_best[::2**(num_approx-j-1)] ) )
	plt.loglog(h,max_error,'.-r',label="$E(h)$")
	plt.loglog(h,h**(2.),'-k',label="$h^{\, 2}$")
	plt.xlabel("$h$")
	plt.legend(loc='best')
	plt.savefig('example_convergence.pdf')
	plt.show()
	print "The order of the finite difference approximation is about ", ( (np.log(max_error[0]) - 
	    np.log(max_error[-1]) )/( np.log(h[0]) - np.log(h[-1]) ) ), "."
	plt.clf()
	return 


def Exercise1():
	def problem1(epsilon=.4,subintervals=20):
	    X,Y = general_secondorder_ode_fd(func=lambda x:-1.,a1=lambda x:epsilon,
	                                    a2=lambda x: -1.,a3=lambda x:0.,
	                                    a=0.,b=1., alpha=1.,beta=3.,N=subintervals)
	    return X,Y
	
	def AnalyticSolution(x,alpha, beta,epsilon):
	    out = alpha+x+(beta-alpha-1.)*(np.exp(x/epsilon) -1.)/(np.exp(1./epsilon) -1.)   
	    return out
	
	
	eps, subintervals = 0.1, 400
	X,Y = problem1(eps, subintervals)
	plt.plot(X,Y,'-k',mfc="None",linewidth=2.0)
	plt.axis([0.,1.1,.8,3.2])
	plt.ylabel('$y$',fontsize=16)
	plt.xlabel('$x$',fontsize=16)
	plt.savefig('figure2.pdf')
	plt.show()
	plt.clf()
	
	
	N = np.array([5,10,20,40,80,160,320,640,1280,2560])
	h, MaxError = 2./N, np.ones(len(N))
	eps = .1
	for j in range(len(N)): 
		Mesh, Sol = problem1(epsilon=eps,subintervals=N[j])
		MaxError[j] = (max(abs( Sol- AnalyticSolution(Mesh,alpha=1.,beta=3.,epsilon=eps) ) ) )
	print "Number of subintervals = ", N
	print "MaxError = ",MaxError
	plt.loglog(h,h**(2.),'-k',label="$h^{\, 2}$" )
	plt.loglog(h,MaxError,'.-r',label="$E(h^2)$")
	plt.xlabel("$h$")
	plt.legend(loc='best')
	# plt.show()
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





# example()
Exercise1()
# ExercisePoisson()


