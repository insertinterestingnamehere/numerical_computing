from __future__ import division
import matplotlib
# matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')


import numpy as np
from solution import general_secondorder_ode_fd, poisson_square
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
import matplotlib.colors as mcolors
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve

def example():
	# First Code block in the lab manual
	import numpy as np
	from scipy.sparse import spdiags
	from scipy.sparse.linalg import spsolve
	
	def bvp(func, a, b, alpha, beta, N):
		# Solving u'' = f(x), x in [a, b],
		# u(a) = alpha, u(b) = beta
		# N = number of subintervals
		#
		# We use the finite difference method to construct a system 
		# of algebraic equations described by the matrix equation 
		# AU = F
		
		h = (b-a)/N 	# The length of each subinterval
		
		# Initialize and define the vector F on the right
		F = np.empty(N-1.)			
		F[0] = func(a+1.*h)-alpha*h**(-2.)
		F[N-2] = func(a+(N-1)*h)-beta*h**(-2.)
		for j in xrange(1,N-2): 
			F[j] = func(a + (j+1)*h)
			
		# Here we define the arrays that will go on the diagonals of A
		data = np.empty((3,N-1))
		data[0,:] = -2.*np.ones((1,N-1)) # main diagonal
		data[1,:], data[2,:] = np.ones((1,N-1)), np.ones((1,N-1))  # off-diagonals
		# Next we specify on which diagonals they will be placed
		diags = np.array([0,-1,1])
		
		A=h**(-2.)*spdiags(data,diags,N-1,N-1).asformat('csr')
		
		# We create and return the numerical approximation
		U = np.empty(N+1)
		U[1:-1] = spsolve(A,F)
		U[0], U[-1] = alpha, beta
		return np.linspace(a,b,N+1), U
	
	x, y = bvp(lambda x:(-3.*np.sin(x)), a=0., b=2., alpha=-2., beta=1, N=30)
	import matplotlib.pyplot as plt
	plt.plot(x,y,'-k',linewidth=2.0)
	plt.show()
	return 


def Exercise1():
	def problem1(epsilon, subintervals):
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


def plotRhos():
	#definitions for atoms position and charges
	#the angle the hydrogen atoms make
	theta = 106.0/180.0*np.pi
	#Length of the two branches
	A = 1.0
	# Hydrogen 1 (x0,y0,q)
	# Hydrogen 2 
	# Oxygen
	water = ((-np.sin(theta/2)*A	, 0				, 1),
		    ( np.sin(theta/2)*A	, 0				 	, 1),
		    ( 0					, -np.cos(theta/2)*A,-2))
	
	def rho1(x,y,atom):
		return atom[2]*np.exp(-np.sqrt((x-atom[0])**2 + (y-atom[1])**2))
	
	def rhoSum(x,y,atoms):
		return np.sum([rho1(x,y,atom) for atom in atoms],axis=0)
	
	#Generate a color dictionary for use with LinearSegmentedColormap
	#that places red and blue at the min and max values of data
	#and white when data is zero
	def genDict(data):
		zero = 1/(1 - np.max(data)/np.min(data))
		cdict = {'red':   [(0.0,  1.0, 1.0),
		               (zero,  1.0, 1.0),
		               (1.0,  0.0, 0.0)],
		     'green': [(0.0,  0.0, 0.0),
		               (zero,  1.0, 1.0),
		               (1.0,  0.0, 0.0)],
		     'blue':  [(0.0,  0.0, 0.0),
		               (zero,  1.0, 1.0),
		               (1.0,  1.0, 1.0)]}
		return cdict
	
	m = 500
	X = np.linspace(-5,5,m)
	X,Y = np.meshgrid(X,X)
	#Generate the grid of rho values
	Rho = rhoSum(X,Y,water)
	plt.imshow(Rho,cmap =  mcolors.LinearSegmentedColormap('cmap', genDict(Rho)))
	plt.colorbar(label="Relative Charge Density")
	plt.savefig("./waterRho.png")
	plt.clf()
	
	co2 = ((-1,0,-2),
        (1,0,-2),
        (0,0,4))
	Rho = rhoSum(X,Y,co2)
	plt.imshow(Rho,cmap =  mcolors.LinearSegmentedColormap('cmap', genDict(Rho)))
	plt.colorbar(label="Relative Charge Density")
	plt.savefig("./co2Rho.png")
	plt.clf()
	return 


def plotVs():
	
	#definitions for atoms position and charges
	#the angle the hydrogen atoms make
	theta = 106.0/180.0*np.pi
	#Length of the two branches
	A = 1.0
	# Hydrogen 1 (x0,y0,q)
	# Hydrogen 2 
	# Oxygen
	water = ((-np.sin(theta/2)*A,0,1),
        (np.sin(theta/2)*A,0,1),
        (0,-np.cos(theta/2)*A,-2))
	
	def rho1(x,y,atom):
		return atom[2]*np.exp(-np.sqrt((x-atom[0])**2 + (y-atom[1])**2))
	
	def rhoSum(x,y,atoms):
		return np.sum([rho1(x,y,atom) for atom in atoms],axis=0)
	
	
	def poisson_square(a1,b1,c1,d1,n,bcs, source): 
		#n = number of subintervals
		# We discretize in the x dimension by a1 = x_0 < x_1< ... < x_n=b1, and 
		# we discretize in the y dimension by c1 = y_0 < y_1< ... < y_n=d1. 
		# This means that we have interior points {x_1, ..., x_{n-1}}\times {y_1, ..., y_{n-1}}
		# or {x_1, ..., x_m}\times {y_1, ..., y_m} where m = n-1. 
		# In Python, this is indexed as {x_0, ..., x_{m-1}}\times {y_0, ..., y_{m-1}}
		# We will have m**2 pairs of interior points, and m**2 corresponding equations.
		# We will organize these equations by their y coordinates: all equations centered 
		# at (x_i, y_0) will be listed first, then (x_i, y_1), and so on till (x_i, y_{m-1})
		delta_x, delta_y, h, m = (b1-a1)/n, (d1-c1)/n, (b1-a1)/n, n-1
		
		####    Here we construct the matrix A    ####
		##############################     Slow            #################################
		#     D, diags = np.ones((1,m**2)), np.array([-m,m])
		#     data = np.concatenate((D, D),axis=0) 
		#     A = h**(-2)*spdiags(data,diags,m**2,m**2).asformat('lil')
		#     D = np.ones((1,m))
		#     diags, data = np.array([0,-1,1]), np.concatenate((-4.*D,D,D),axis=0)
		#     temp = h**(-2)*spdiags(data,diags,m,m).asformat('lil')
		#     for i in xrange(m): A[i*m:(i+1)*m,i*m:(i+1)*m] = temp
		
		##############################     Much Faster      ################################
		D1,D2,D3 = -4*np.ones((1,m**2)), np.ones((1,m**2)), np.ones((1,m**2)) 
		Dm1, Dm2 = np.ones((1,m**2)), np.ones((1,m**2))
		for j in range(0,D2.shape[1]):
			if (j%m)==m-1: D2[0,j]=0
			if (j%m)==0: D3[0,j]=0
		diags = np.array([0,-1,1,-m,m])
		data = np.concatenate((D1,D2,D3,Dm1,Dm2),axis=0) # This stacks up rows
		A = 1./h**2.*spdiags(data, diags, m**2,m**2).asformat('csr') # This appears to work correctly
		
		####    Here we construct the vector b    ####
		b, Array = np.zeros(m**2), np.linspace(0.,1.,m+2)[1:-1]
		# In the next line, source represents the inhomogenous part of Poisson's equation
		for j in xrange(m): b[j*m:(j+1)*m] = source(a1+(b1-a1)*Array, c1+(j+1)*h*np.ones(m) )
		
		# In the next four lines, bcs represents the Dirichlet conditions on the boundary
	#     y = c1+h, d1-h
		b[0:m] = b[0:m] - h**(-2.)*bcs(a1+(b1-a1)*Array,c1*np.ones(m))
		b[(m-1)*m : m**2] = b[(m-1)*m : m**2] - h**(-2.)*bcs(a1+(b1-a1)*Array,d1*np.ones(m))
	#     x = a1+h, b1-h
		b[0::m] = b[0::m] - h**(-2.)*bcs(a1*np.ones(m),c1+(d1-c1)*Array) 
		b[(m-1)::m] = b[(m-1)::m] - h**(-2.)*bcs(b1*np.ones(m),c1+(d1-c1)*Array)
		
		####    Here we solve the system A*soln = b    ####
		soln = spsolve(A,b) # Using the conjugate gradient method: (soln, info) = cg(A,b)
		
		z = np.zeros((m+2,m+2) ) 
		for j in xrange(m): z[1:-1,j+1] = soln[j*m:(j+1)*m]
		
		x, y = np.linspace(a1,b1,m+2), np.linspace(c1,d1,m+2)
		z[:,0], z[:,m+1]  = bcs(x,c1*np.ones(len(x)) ), bcs(x,d1*np.ones(len(x)) )
		z[0,:], z[m+1,:] = bcs(a1*np.ones(len(x)),y), bcs(b1*np.ones(len(x)),y)
		return z
	
	#Generate a color dictionary for use with LinearSegmentedColormap
	#that places red and blue at the min and max values of data
	#and white when data is zero
	def genDict(data):
		zero = 1/(1 - np.max(data)/np.min(data))
		cdict = {'red':   [(0.0,  1.0, 1.0),
		               (zero,  1.0, 1.0),
		               (1.0,  0.0, 0.0)],
		     'green': [(0.0,  0.0, 0.0),
		               (zero,  1.0, 1.0),
		               (1.0,  0.0, 0.0)],
		     'blue':  [(0.0,  0.0, 0.0),
		               (zero,  1.0, 1.0),
		               (1.0,  1.0, 1.0)]}
		return cdict
	
	a1 = -5
	b1 = 5
	m = 500
	X = np.linspace(a1,b1,m)
	X,Y = np.meshgrid(X,X)
	#poisson_square seems to mix up x and y
	Rho = poisson_square(a1,b1,a1,b1,m,lambda x,y:0 , lambda x,y: -rhoSum(y,x,water))
	cdict = genDict(Rho)
	
	plt.imshow(Rho,cmap = mcolors.LinearSegmentedColormap('CustomMap', cdict))
	plt.colorbar(label="Relative Voltage")
	plt.savefig("./waterV.png")
	plt.clf()
	
	co2 = ((-1,0,-2),
		    (1,0,-2),
		    (0,0,4))
	Rho = poisson_square(a1,b1,a1,b1,m,lambda x,y:0 , lambda x,y: -rhoSum(y,x,co2))
	cdict = genDict(Rho)
	plt.imshow(Rho,cmap = mcolors.LinearSegmentedColormap('CustomMap', cdict))
	plt.colorbar(label="Relative Voltage")
	plt.savefig("./co2V.png")
	plt.clf()
	return 



# example()
# Exercise1()
# ExercisePoisson()
# plotRhos()
plotVs()

