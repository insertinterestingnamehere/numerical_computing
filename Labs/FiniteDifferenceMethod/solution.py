from __future__ import division
import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve, cg
import matplotlib.pyplot as plt

def fd_order2_ode(func,a1,a2,a3,a=0.,b=1.,alpha=1.,beta=3.,N=5):
	# A Simple Finite Difference Scheme to solve BVP's of the form 
	# a1(x)u''(x) + a2(x)u'(x) + a3(x)u(x) = f(x), x \in [a,b]
	# u(a) = alpha
	# u(b) = beta
	# (Dirichlet boundary conditions)
	# 
	# U_0 = alpha, U_1, U_2, ..., U_m, U_{m+1} = beta
	# We use m+1 subintervals, giving m algebraic equations
	m = N-1
	h = (b-a)/N		  # Here we form the diagonals
	x = np.linspace(a,b,N+1)
	D0,Dp,Dm,diags = np.zeros((1,m)), np.zeros((1,m)), np.zeros((1,m)), np.array([0,-1,1])
	
	D0 += -2.*a1(x[1:-1])*h**(-2.) + a3(x[1:-1])
	Dm += a1(x[1:-1])*h**(-2.) - a2(x[1:-1])*(2.*h)**(-1.)
	Dp += a1(x[1:-1])*h**(-2.) + a2(x[1:-1])*(2.*h)**(-1.)
	# print "\nD0 = \n", D0[0,:5]
	# print "\nDm = \n", Dm[0,:5]
	# print "\nDp = \n", Dp[0,:5]
	# Here we create the matrix A
	data = np.concatenate((D0,np.roll(Dm,-1),np.roll(Dp,1)),axis=0) # This stacks up rows
	A = spdiags(data,diags,m,m).asformat('csr')
	# print "\nA = \n", A[:5,:5].todense()
	# print "\nA = \n", A[-5:,-5:].todense()
	
	# Here we create the vector B
	B = np.zeros(N+1)
	B[2:-2] = func(x[2:-2])	
	xj = a+1.*h
	B[0], B[1] = alpha, func(xj)-alpha *( a1(xj)*h**(-2.) - a2(xj)*(2.*h)**(-1.) )
	xj = a+m*h
	B[-1], B[-2]  = beta, func(xj)-beta*( a1(xj)*h**(-2.) + a2(xj)*(2.*h)**(-1.) )
	# print "\nB = \n", B[:5]
	# print "\nB = \n", B[-5:]
	
	# Here we solve the equation AX = B and return the result
	B[1:-1] = spsolve(A,B[1:-1])
	return np.linspace(a,b,m+2), B



def approx_order(num_approx,N,bvp,*args):
	h, max_error = (1.-0)/N[:-1], np.ones(num_approx-1)
	
	mesh_best, num_sol_best = bvp(*args, subintervals=N[-1])
	for j in range(len(N)-1): 
		mesh, num_sol = bvp(*args, subintervals=N[j])
		max_error[j] = np.max(np.abs( num_sol- num_sol_best[::2**(num_approx-j-1)] ) )
	plt.loglog(h,max_error,'.-r',label="$E(h)$")
	plt.loglog(h,h**(2.),'-k',label="$h^{\, 2}$")
	plt.xlabel("$h$")
	plt.legend(loc='best')
	plt.show()
	print "The order of the finite difference approximation is about ", ( (np.log(max_error[0]) - 
		np.log(max_error[-1]) )/( np.log(h[0]) - np.log(h[-1]) ) ), "."



# 
# def example():
#	# First Code block in the lab manual
#	import numpy as np
#	from scipy.sparse import spdiags
#	from scipy.sparse.linalg import spsolve
#	
#	def bvp(func, epsilon, alpha, beta, N):
#		a,b = 0., 1.	# Interval for the BVP
#		h = (b-a)/N		# The length of each subinterval
#		
#		# Initialize and define the vector F on the right
#		F = np.empty(N-1.)			
#		F[0] = func(a+1.*h)-alpha*(epsilon+h/2.)*h**(-2.)
#		F[N-2] = func(a+(N-1)*h)-beta*(epsilon-h/2.)*h**(-2.)
#		for j in xrange(1,N-2): 
#			F[j] = func(a + (j+1)*h)
#		
#		# Here we define the arrays that will go on the diagonals of A
#		data = np.empty((3,N-1))
#		data[0,:] = -2.*epsilon*np.ones((1,N-1)) # main diagonal
#		data[1,:]  = (epsilon+h/2.)*np.ones((1,N-1))	 # off-diagonals
#		data[2,:] = (epsilon-h/2.)*np.ones((1,N-1))
#		# Next we specify on which diagonals they will be placed, and create A
#		diags = np.array([0,-1,1])
#		A=h**(-2.)*spdiags(data,diags,N-1,N-1).asformat('csr')
#		
#		U = np.empty(N+1)
#		U[1:-1] = spsolve(A,F)
#		U[0], U[-1] = alpha, beta
#		return np.linspace(a,b,N+1), U
#	
#	x, y = bvp(lambda x:-1., epsilon=.05,alpha=1, beta=3, N=400)
#	import matplotlib.pyplot as plt
#	plt.plot(x,y,'-k',linewidth=2.0)
#	plt.show()
#	
#	num_approx = 5 # Number of Approximations
#	N = 20*np.array([2**j for j in range(num_approx)])
#	h, max_error = (1.-0)/N[:-1], np.ones(num_approx-1)
#	
#	mesh_best, num_sol_best = bvp(lambda x:-1, epsilon=.5, alpha=1, beta=3, N=N[-1])
#	for j in range(len(N)-1): 
#		mesh, num_sol = bvp(lambda x:-1, epsilon=.5, alpha=1, beta=3, N=N[j])
#		max_error[j] = np.max(np.abs( num_sol- num_sol_best[::2**(num_approx-j-1)] ) )
#	plt.loglog(h,max_error,'.-r',label="$E(h)$")
#	plt.loglog(h,h**(2.),'-k',label="$h^{\, 2}$")
#	plt.xlabel("$h$")
#	plt.legend(loc='best')
#	plt.show()
#	print "The order of the finite difference approximation is about ", ( (np.log(max_error[0]) - 
#		np.log(max_error[-1]) )/( np.log(h[0]) - np.log(h[-1]) ) ), "."
#	return 
# 
