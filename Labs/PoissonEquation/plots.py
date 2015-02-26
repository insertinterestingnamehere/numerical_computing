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
from solution import poisson_square


def ExercisePoisson():
	from numpy import sin, cos, pi
	# Domain: [0,1]x[0,1]
	a1,b1 = 0.,1.
	c1,d1 = 0.,1.
	n=100
	# Example1: Laplace's equation (Poisson with no source)
	def bcs(x,y): 
		return x**3.
	
	def source(x,y): 
		return 0.
	
	
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
	#			linewidth=0, antialiased=False)
	print "Maximum Error = \n", np.max(np.abs( z-bcs(xv,yv) ) )
	# plt.savefig('Laplace.pdf')
	# plt.clf()
	plt.show()
	
	
	# if True: return
	# 
	# num_approx = 7 # Number of Approximations
	# N = np.array([10*2**(j) for j in range(num_approx)])
	# h, max_error = (b1-a1)/N[:-1], np.ones(num_approx-1)
	# 
	# num_sol_best = poisson_square(a1,b1,c1,d1,N[-1],bcs,source)
	# for j in range(len(N)-1): 
	# 	num_sol = poisson_square(a1,b1,c1,d1,N[j],bcs,source)
	# 	max_error[j] = np.max(np.abs( num_sol- num_sol_best[::2**(num_approx-j-1), ::2**(num_approx-j-1)] ) )
	# plt.loglog(h,max_error,'.-r',label="$E(h)$")
	# plt.loglog(h,h**(2.),'-k',label="$h^{\, 2}$")
	# plt.xlabel("$h$")
	# plt.legend(loc='best')
	# print "The order of the finite difference approximation is about ", ( (np.log(max_error[0]) - 
	# 	np.log(max_error[-1]) )/( np.log(h[0]) - np.log(h[-1]) ) ), "."
	# plt.savefig('./Poisson_Error.pdf')
	# plt.show()
	return



def plotRhos():
	def source(X,Y):
		"""
		Takes arbitrary arrays of coordinates X and Y and returns an array of the same shape
		representing the charge density of nested charged squares
		"""
		src = np.zeros(X.shape)
		src[ np.logical_or(
			np.logical_and( np.logical_or(abs(X-1.5) < .1,abs(X+1.5) < .1) ,abs(Y) < 1.6),
			np.logical_and( np.logical_or(abs(Y-1.5) < .1,abs(Y+1.5) < .1) ,abs(X) < 1.6))] = 1
		src[ np.logical_or(
			np.logical_and( np.logical_or(abs(X-0.9) < .1,abs(X+0.9) < .1) ,abs(Y) < 1.0),
			np.logical_and( np.logical_or(abs(Y-0.9) < .1,abs(Y+0.9) < .1) ,abs(X) < 1.0))] = -1
		return src
	
	#Generate a color dictionary for use with LinearSegmentedColormap
	#that places red and blue at the min and max values of data
	#and white when data is zero
	def genDict(data):
		zero = 1/(1 - np.max(data)/np.min(data))
		cdict = {'red':	  [(0.0,  1.0, 1.0),
					(zero,	1.0, 1.0),
					(1.0,  0.0, 0.0)],
			'green': [(0.0,	 0.0, 0.0),
					(zero,	1.0, 1.0),
					(1.0,  0.0, 0.0)],
			'blue':	 [(0.0,	 0.0, 0.0),
					(zero,	1.0, 1.0),
					(1.0,  1.0, 1.0)]}
		return cdict
	
	
	a1 = -2.
	b1 = 2.
	c1 = -2.
	d1 = 2.
	n =100
	X = np.linspace(a1,b1,n)
	Y = np.linspace(c1,d1,n)
	X,Y = np.meshgrid(X,Y)
	
	rho= source(X,Y)
	
	plt.imshow(rho,cmap =  mcolors.LinearSegmentedColormap('cmap', genDict(rho)))
	plt.colorbar(label="Relative Charge")
	plt.show()
	# plt.savefig("./pipesRho.pdf")
	plt.clf()
	return



def plotVs():
	
# 	
# 	def poisson_square(a1,b1,c1,d1,n,bcs, source): 
# 		#n = number of subintervals
# 		# We discretize in the x dimension by a1 = x_0 < x_1< ... < x_n=b1, and 
# 		# we discretize in the y dimension by c1 = y_0 < y_1< ... < y_n=d1. 
# 		# This means that we have interior points {x_1, ..., x_{n-1}}\times {y_1, ..., y_{n-1}}
# 		# or {x_1, ..., x_m}\times {y_1, ..., y_m} where m = n-1. 
# 		# In Python, this is indexed as {x_0, ..., x_{m-1}}\times {y_0, ..., y_{m-1}}
# 		# We will have m**2 pairs of interior points, and m**2 corresponding equations.
# 		# We will organize these equations by their y coordinates: all equations centered 
# 		# at (x_i, y_0) will be listed first, then (x_i, y_1), and so on till (x_i, y_{m-1})
# 		delta_x, delta_y, h, m = (b1-a1)/n, (d1-c1)/n, (b1-a1)/n, n-1
# 		
# 		####	Here we construct the matrix A	  ####
# 		##############################	   Slow			   #################################
# 		#	  D, diags = np.ones((1,m**2)), np.array([-m,m])
# 		#	  data = np.concatenate((D, D),axis=0) 
# 		#	  A = h**(-2)*spdiags(data,diags,m**2,m**2).asformat('lil')
# 		#	  D = np.ones((1,m))
# 		#	  diags, data = np.array([0,-1,1]), np.concatenate((-4.*D,D,D),axis=0)
# 		#	  temp = h**(-2)*spdiags(data,diags,m,m).asformat('lil')
# 		#	  for i in xrange(m): A[i*m:(i+1)*m,i*m:(i+1)*m] = temp
# 		
# 		##############################	   Much Faster		################################
# 		D1,D2,D3 = -4*np.ones((1,m**2)), np.ones((1,m**2)), np.ones((1,m**2)) 
# 		Dm1, Dm2 = np.ones((1,m**2)), np.ones((1,m**2))
# 		for j in range(0,D2.shape[1]):
# 				if (j%m)==m-1: D2[0,j]=0
# 				if (j%m)==0: D3[0,j]=0
# 		diags = np.array([0,-1,1,-m,m])
# 		data = np.concatenate((D1,D2,D3,Dm1,Dm2),axis=0) # This stacks up rows
# 		A = 1./h**2.*spdiags(data, diags, m**2,m**2).asformat('csr') # This appears to work correctly
# 		
# 		####	Here we construct the vector b	  ####
# 		b, Array = np.zeros(m**2), np.linspace(0.,1.,m+2)[1:-1]
# 		# In the next line, source represents the inhomogenous part of Poisson's equation
# 		for j in xrange(m): b[j*m:(j+1)*m] = source(a1+(b1-a1)*Array, c1+(j+1)*h*np.ones(m) )
# 		
# 		# In the next four lines, bcs represents the Dirichlet conditions on the boundary
# #	  y = c1+h, d1-h
# 		b[0:m] = b[0:m] - h**(-2.)*bcs(a1+(b1-a1)*Array,c1*np.ones(m))
# 		b[(m-1)*m : m**2] = b[(m-1)*m : m**2] - h**(-2.)*bcs(a1+(b1-a1)*Array,d1*np.ones(m))
# #	  x = a1+h, b1-h
# 		b[0::m] = b[0::m] - h**(-2.)*bcs(a1*np.ones(m),c1+(d1-c1)*Array) 
# 		b[(m-1)::m] = b[(m-1)::m] - h**(-2.)*bcs(b1*np.ones(m),c1+(d1-c1)*Array)
# 		
# 		####	Here we solve the system A*soln = b	   ####
# 		soln = spsolve(A,b) # Using the conjugate gradient method: (soln, info) = cg(A,b)
# 		
# 		z = np.zeros((m+2,m+2) ) 
# 		for j in xrange(m): z[1:-1,j+1] = soln[j*m:(j+1)*m]
# 		
# 		x, y = np.linspace(a1,b1,m+2), np.linspace(c1,d1,m+2)
# 		z[:,0], z[:,m+1]  = bcs(x,c1*np.ones(len(x)) ), bcs(x,d1*np.ones(len(x)) )
# 		z[0,:], z[m+1,:] = bcs(a1*np.ones(len(x)),y), bcs(b1*np.ones(len(x)),y)
# 		return z
# 	
# 	
	def source(X,Y):
		"""
		Takes arbitrary arrays of coordinates X and Y and returns an array of the same shape
		representing the charge density of nested charged squares
		"""
		src = np.zeros(X.shape)
		src[ np.logical_or(
			np.logical_and( np.logical_or(abs(X-1.5) < .1,abs(X+1.5) < .1) ,abs(Y) < 1.6),
			np.logical_and( np.logical_or(abs(Y-1.5) < .1,abs(Y+1.5) < .1) ,abs(X) < 1.6))] = 1
		src[ np.logical_or(
			np.logical_and( np.logical_or(abs(X-0.9) < .1,abs(X+0.9) < .1) ,abs(Y) < 1.0),
			np.logical_and( np.logical_or(abs(Y-0.9) < .1,abs(Y+0.9) < .1) ,abs(X) < 1.0))] = -1
		return src

	#Generate a color dictionary for use with LinearSegmentedColormap
	#that places red and blue at the min and max values of data
	#and white when data is zero
	def genDict(data):
		zero = 1/(1 - np.max(data)/np.min(data))
		cdict = {'red':	  [(0.0,  1.0, 1.0),
					(zero,	1.0, 1.0),
					(1.0,  0.0, 0.0)],
			'green': [(0.0,	 0.0, 0.0),
					(zero,	1.0, 1.0),
					(1.0,  0.0, 0.0)],
			'blue':	 [(0.0,	 0.0, 0.0),
					(zero,	1.0, 1.0),
					(1.0,  1.0, 1.0)]}
		return cdict



	a1 = -2.
	b1 = 2.
	c1 = -2.
	d1 = 2.
	n = 5
	# X = np.linspace(a1,b1,n)
	# Y = np.linspace(c1,d1,n)
	# X,Y = np.meshgrid(X,Y)
	# 
	# rho= source(X,Y)
	V = poisson_square(a1,b1,c1,d1,6,lambda x, y:0, lambda X,Y: source(X,Y))
	cdict = genDict(V)

	plt.imshow(V,cmap = mcolors.LinearSegmentedColormap('CustomMap', cdict))
	plt.colorbar(label="Voltage")
	plt.show()
	# plt.savefig("./pipesV.pdf")
	plt.clf()
	# print X.shape
	# print Y.shape
	print V.shape
	return

if __name__ == "__main__":
	# example()
	# Exercise1()
	# ExercisePoisson()
	# plotRhos()
	plotVs()

