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
from solution import general_secondorder_ode_fd

def example():
	# First Code block in the lab manual
	import numpy as np
	from scipy.sparse import spdiags
	from scipy.sparse.linalg import spsolve
	
	def bvp(func, epsilon, alpha, beta, N):
		a,b = 0., 1.	# Interval for the BVP
		h = (b-a)/N		# The length of each subinterval
		
		# Initialize and define the vector F on the right
		F = np.empty(N-1.)			
		F[0] = func(a+1.*h)-alpha*(epsilon+h/2.)*h**(-2.)
		F[N-2] = func(a+(N-1)*h)-beta*(epsilon-h/2.)*h**(-2.)
		for j in xrange(1,N-2): 
			F[j] = func(a + (j+1)*h)
		
		# Here we define the arrays that will go on the diagonals of A
		data = np.empty((3,N-1))
		data[0,:] = -2.*epsilon*np.ones((1,N-1)) # main diagonal
		data[1,:]  = (epsilon+h/2.)*np.ones((1,N-1))	 # off-diagonals
		data[2,:] = (epsilon-h/2.)*np.ones((1,N-1))
		# Next we specify on which diagonals they will be placed, and create A
		diags = np.array([0,-1,1])
		A=h**(-2.)*spdiags(data,diags,N-1,N-1).asformat('csr')
		
		U = np.empty(N+1)
		U[1:-1] = spsolve(A,F)
		U[0], U[-1] = alpha, beta
		return np.linspace(a,b,N+1), U
	
	x, y = bvp(lambda x:-1., epsilon=.05,alpha=1, beta=3, N=400)
	import matplotlib.pyplot as plt
	plt.plot(x,y,'-k',linewidth=2.0)
	plt.show()
	
	num_approx = 5 # Number of Approximations
	N = 20*np.array([2**j for j in range(num_approx)])
	h, max_error = (1.-0)/N[:-1], np.ones(num_approx-1)
	
	mesh_best, num_sol_best = bvp(lambda x:-1, epsilon=.5, alpha=1, beta=3, N=N[-1])
	for j in range(len(N)-1): 
		mesh, num_sol = bvp(lambda x:-1, epsilon=.5, alpha=1, beta=3, N=N[j])
		max_error[j] = np.max(np.abs( num_sol- num_sol_best[::2**(num_approx-j-1)] ) )
	plt.loglog(h,max_error,'.-r',label="$E(h)$")
	plt.loglog(h,h**(2.),'-k',label="$h^{\, 2}$")
	plt.xlabel("$h$")
	plt.legend(loc='best')
	plt.show()
	print "The order of the finite difference approximation is about ", ( (np.log(max_error[0]) - 
		np.log(max_error[-1]) )/( np.log(h[0]) - np.log(h[-1]) ) ), "."
	return 


def Exercise1():
	def bvp(epsilon, subintervals):
		X,Y = general_secondorder_ode_fd(func=lambda x:0.,a1=lambda x:epsilon,
										  a2=lambda x: x,a3=lambda x:-x,
										  a=0.,b=1., alpha=0.,beta=np.exp(1.),N=subintervals)
		
		# for figure2.pdf
		# X,Y = general_secondorder_ode_fd(func=lambda x:-1.,a1=lambda x:epsilon,
		# 								a2=lambda x:-1.,a3=lambda x:0.,
		# 								a=0.,b=1., alpha=1.,beta=3.,N=subintervals)
		
		# for figure3.pdf
		# X,Y = general_secondorder_ode_fd(func=lambda x:np.cos(x),a1=lambda x:epsilon,
		# 								  a2=lambda x: 0.,a3=lambda x:-4.*(np.pi-x**2.),
		# 								  a=0.,b=np.pi/2., alpha=0.,beta=1.,N=subintervals)
		return X,Y
	
	
	# def AnalyticSolution(x,alpha, beta,epsilon):
	#	  out = alpha+x+(beta-alpha-1.)*(np.exp(x/epsilon) -1.)/(np.exp(1./epsilon) -1.)   
	#	  return out
	
	
	eps, subintervals = 0.1, 400
	X,Y = bvp(eps, subintervals)
	plt.plot(X,Y,'-k',mfc="None",linewidth=2.0)
	plt.ylabel('$y$',fontsize=16)
	plt.xlabel('$x$',fontsize=16)
	# plt.axis([-.1,1.1,1-.1,3+.1])
	# plt.savefig('figure2.pdf')
	plt.axis([-.1,np.pi/2.+.1,-.1,1.5])
	# plt.savefig('figure3.pdf')
	plt.show()
	plt.clf()
	
	num_approx = 10 # Number of Approximations
	N = 20*np.array([2**j for j in range(num_approx)])
	h, max_error = (1.-0)/N[:-1], np.ones(num_approx-1)

	mesh_best, num_sol_best = bvp(eps, subintervals=N[-1])
	for j in range(len(N)-1): 
		mesh, num_sol = bvp(eps, subintervals=N[j])
		max_error[j] = np.max(np.abs( num_sol- num_sol_best[::2**(num_approx-j-1)] ) )
	plt.loglog(h,max_error,'.-r',label="$E(h)$")
	plt.loglog(h,h**(2.),'-k',label="$h^{\, 2}$")
	plt.xlabel("$h$")
	plt.legend(loc='best')
	plt.show()
	print "The order of the finite difference approximation is about ", ( (np.log(max_error[5]) - 
		np.log(max_error[-1]) )/( np.log(h[5]) - np.log(h[-1]) ) ), "."
	# N = np.array([5,10,20,40,80,160,320,640,1280,2560])
	# h, MaxError = 2./N, np.ones(len(N))
	# eps = .1
	# for j in range(len(N)): 
	#	Mesh, Sol = problem1(epsilon=eps,subintervals=N[j])
	#	MaxError[j] = (max(abs( Sol- AnalyticSolution(Mesh,alpha=1.,beta=3.,epsilon=eps) ) ) )
	# print "Number of subintervals = ", N
	# print "MaxError = ",MaxError
	# plt.loglog(h,h**(2.),'-k',label="$h^{\, 2}$" )
	# plt.loglog(h,MaxError,'.-r',label="$E(h^2)$")
	# plt.xlabel("$h$")
	# plt.legend(loc='best')
	# # plt.show()
	# print "Order of the Approximation is about ", ( (np.log(MaxError[0]) - 
	#				  np.log(MaxError[-1]) )/( np.log(h[0]) - np.log(h[-1]) ) )
	return 



# example()
Exercise1()


