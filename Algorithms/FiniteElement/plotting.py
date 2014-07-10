from __future__ import division

import numpy as np
from numpy import array as arr
from scipy.interpolate import interp1d
# from scipy.sparse.linalg import spsolve
# from scipy.sparse import spdiags, coo_matrix
import matplotlib.pyplot as plt
from matplotlib.axis import Axis

from solution import ode_fe
from tridiag import tridiag

def Example3(N,epsilon):
	x, y = ode_fe(func=lambda x:-1./epsilon,c=-1./epsilon,x=np.linspace(0,1,N))
	plt.plot(x,y,'-ro')
	plt.axis([0.,1.1,.8,3.2])
	plt.show()
	return x, y


def NonlinearGrid():
	epsilon = .01
	N=1000
	X = np.linspace(0,1,N)
	x1, y1 = ode_fe(func=lambda x:-1./epsilon,c=-1./epsilon,x=X)
	x2, y2 = ode_fe(func=lambda x:-1./epsilon,c=-1./epsilon,x=X**(1./14.))
	alpha, beta = 1.,3.
	# Analytic solution
	def asol(x,epsilon):
		A = []
		for item in x:
			A.append(alpha+item + (beta-alpha-1.)*(np.exp(item/epsilon) -1.)/(np.exp(1./epsilon) -1.)   )
		return x, np.array(A)
	
	
	
	Z =asol(np.linspace(0,1,500),epsilon)
	
	plt.plot(x1,y1,'-bo')
	plt.plot(Z[0],Z[1],'-k',mfc="None")
	plt.plot(x2,y2,'-ro')
	plt.axis([0.,1.1,.8,3.2])
	plt.show()
	plt.clf()
	
	X = np.linspace(0,1,500)
	plt.plot(X,abs(Z[1]-interp1d(x2,y2)(X)),'-r')
	plt.plot( X,abs( Z[1]-interp1d(x1,y1)(X) ), '-b' )
	
	print "Max Error = ", np.max(np.abs(Z[1]-interp1d(x2,y2)(X) ))
	plt.show()
	return 


def tridiagonal_approach():
	epsilon, alpha, beta = .02, 1., 3.
	N = 5 # subintervals or finite elements
	x = np.linspace(0,1,N+1)**(1./8) # N+1 = number of grid points
	h = np.diff(x) # h[i] = 
	b, f = np.zeros(N+1), np.zeros(N+1)
	a, c= np.zeros(N), np.zeros(N)
	b[0], b[N] = 1., 1.
	f[0], f[N] = alpha, beta
	for i in range(0,N-1):
		b[i+1] = -epsilon*(1./h[i] + 1./h[i+1])# a(\phi_i, \phi_i) 
		f[i+1] = -.5*(h[i] + h[i+1])
		a[i] = epsilon/h[i] + .5
		c[i+1] = epsilon/h[i+1] - .5
		
	# print "a = \n", a
	# print "b = \n", b
	# print "c = \n", c
	# print "f = \n", f
	# print "x = \n", x
	numerical_soln = tridiag(a,b,c,f)
	
	
	def AnalyticSolution(x,alpha, beta,epsilon):
	    out = alpha+x+(beta-alpha-1.)*(np.exp(x/epsilon) -1.)/(np.exp(1./epsilon) -1.)   
	    return out
	
	analytic_soln = AnalyticSolution(x,alpha, beta,epsilon)
	print "Error = ", np.max(np.abs(numerical_soln- analytic_soln))
	plt.plot(x,numerical_soln,'-r',lw=2.)
	plt.plot(x,analytic_soln,'-*k',lw=2.)
	plt.show()
	
	
	return 


def basis_functions_plot():
	fig = plt.figure()
	ax = fig.add_subplot(111)
	# ax = fig.add_axes([-.1,1.1, -.1,1.1])
	
	# Basis function phi_0
	x, y = arr([0,.2,.4,1.]), arr([1.,0.,0.,0.])
	ax.plot(x,y,'-k',linewidth=2.0)
	ax.text(0.,1.02,r"$\phi_0$",fontsize=18)
	
	# Basis function phi_1
	x, y = arr([0,.2,.4,1.]), arr([0.,1.,0.,0.])
	ax.plot(x,y,color='g',lw=2.0)
	ax.text(0.2,1.02,r"$\phi_1$",fontsize=18)
	
	# Basis function phi_2
	x, y = arr([0,.2,.4,.6,1.]), arr([0.,0.,1.,0.,0.])
	ax.plot(x,y,color='r',lw=2.0)
	ax.text(0.4,1.02,r"$\phi_2$",fontsize=18)
	
	# Basis function phi_3
	x, y = arr([0,.4,.6,.8,1.]), arr([0.,0.,1.,0.,0.])
	ax.plot(x,y,'-m',linewidth=2.0)
	ax.text(0.6,1.02,r"$\phi_3$",fontsize=18)
	
	# Basis function phi_4
	x, y = arr([0,.6,.8,1.]), arr([0.,0.,1.,0.])
	ax.plot(x,y,'-b',linewidth=2.0)
	ax.text(0.8,1.02,r"$\phi_4$",fontsize=18)
	
	# Basis function phi_5
	x, y = arr([0,.8,1.]), arr([0.,0.,1.])
	ax.plot(x,y,'-c',linewidth=2.0)
	ax.text(1.,1.02,r"$\phi_5$",fontsize=18)
	
	ax.set_ylim(-.1,1.1)
	ax.set_xlim(-.1,1.1)
	plt.xticks(np.linspace(0.,1.,6),['$x_0$','$x_1$','$x_2$','$x_3$','$x_4$','$x_5$'],fontsize=18)
	# print Axis.get_majorticklabels(plt.axis)
	plt.savefig("basis_functions.pdf")
	
	plt.clf()
	return 


def one_basis_function_plot():
	fig = plt.figure()
	ax = fig.add_subplot(111)
	# ax = fig.add_axes([-.1,1.1, -.1,1.1])
	
	# Basis function phi_3
	x, y = arr([0,.4,.6,.8,1.]), arr([0.,0.,1.,0.,0.])
	ax.plot(x,y,'-m',linewidth=2.0)
	ax.text(0.6,1.02,r"$\phi_3$",fontsize=18)
	
	
	ax.set_ylim(-.1,1.1)
	ax.set_xlim(-.1,1.1)
	# plt.xticks(np.linspace(0.,1.,6),['$x_0$','$x_1$','$x_2$','$x_3$','$x_4$','$x_5$'],fontsize=18)
	plt.xticks(arr([.4,.6,.8]),['$x_2$','$x_3$','$x_4$'],fontsize=18)
	# print Axis.get_majorticklabels(plt.axis)
	plt.savefig("one_basis_function.pdf")
	
	# plt.show()
	plt.clf()
	return 





# Example3(N=100,epsilon=.01)
# NonlinearGrid()
# basis_functions_plot()
# one_basis_function_plot()
tridiagonal_approach()