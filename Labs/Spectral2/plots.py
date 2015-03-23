from __future__ import division
import matplotlib
# matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import numpy as np
from numpy import mod
from numpy.linalg import solve
from scipy.fftpack import fft, ifft
from solution import cheb
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import math

from time import time
from scipy.optimize import root
from scikits import bvp_solver


# Uses bvp_solver from scikits

def deriv1():
	N = 100
	x = (2.*np.pi/N)*np.arange(1,N+1)
	k = np.concatenate(( np.arange(0,N/2) ,
						 np.array([0])	,
						 np.arange(-N/2+1,0,1)	))
	
	k2 = np.concatenate(( np.arange(0,N/2+1) ,
						 # np.array([0])	,
						 np.arange(-N/2+1,0,1)	))
	
	v = np.sin(x)**2.*np.cos(x) + np.exp(2.*np.sin(x+1)) 
	analytic_vp = 2.*np.sin(x)*np.cos(x)**2. - np.sin(x)**3. + 2*np.cos(x+1)*np.exp(2*np.sin(x+1))
	
	analytic_vpp = (2*np.cos(x)**3. -
					4*np.sin(x)**2.*np.cos(x) - 
					3*np.sin(x)**2*np.cos(x)-
					2*np.sin(x+1)*np.exp(2*np.sin(x+1)) + 
					4*np.cos(x+1)**2*np.exp(2*np.sin(x+1))
					)
					
	v_hat = fft(v)
	vp_hat = ((1j*k)*v_hat)
	vp = np.real(ifft(vp_hat))
	
	vpp_hat = ((1j*k2)**2.*v_hat)
	vpp = np.real(ifft(vpp_hat))
	
	# plt.plot(x,analytic_vp,'-g',linewidth=2.)
	# plt.plot(x,vp,'-b',linewidth=2.)
	# plt.show()
	# plt.clf()
	
	
	numerical_solution = .5*analytic_vpp - analytic_vp
	# plt.plot(x,analytic_vpp,'-g',linewidth=2.)
	# plt.plot(x,vpp,'-b',linewidth=2.)
	# plt.show()
	
	plt.plot(x,numerical_solution,'-g',linewidth=2.)
	# plt.plot(x,vpp,'-b',linewidth=2.)
	plt.show()
	return 



def plot_spectral2_derivative():
	N=24
	x1 = (2.*np.pi/N)*np.arange(1,N+1)
	v = np.sin(x1)**2.*np.cos(x1) + np.exp(2.*np.sin(x1+1))
	
	
	k = np.concatenate(( np.arange(0,N/2) ,
						 np.array([0])	, # Because w_hat at N/2 is zero
						 np.arange(-N/2+1,0,1)	))
						
	# Approximates the derivative using the pseudospectral method
	v_hat = fft(v)
	vp_hat = ((1j*k)*v_hat)
	vp = np.real(ifft(vp_hat))
	
	# Calculates the derivative analytically
	x2 = np.linspace(0,2*np.pi,200)
	derivative = (2.*np.sin(x2)*np.cos(x2)**2. - 
					np.sin(x2)**3. + 
					2*np.cos(x2+1)*np.exp(2*np.sin(x2+1))
					)
					
	plt.plot(x2,derivative,'-k',linewidth=2.)
	plt.plot(x1,vp,'*b')
	# plt.savefig('spectral2_derivative.pdf')
	# plt.show()
	# plt.clf()
	return


def advection():
	# variable coefficient wave equation
	# Borrowed from p6.m in Spectral Methods in MATLAB, by Lloyd Trefethen
	
	# Grid, variable coefficient, and initial data:
	N = 128
	h = 2.*np.pi/N
	x = h*np.arange(1,N+1)
	t = 0.
	dt = h/4.
	c = .2 + np.sin(x-1)**2.
	v = np.exp(-100*(x-1)**2)
	vold = np.exp(-100.*(x-.2*dt-1.)**2.)
	k = np.concatenate(( np.arange(0,N/2) ,
						 np.array([0])	,
						 np.arange(-N/2+1,0,1)	))
	
	# Time-stepping with RK:
	tmax = 8
	tplot = .15
	plotgap = int(round(tplot/dt))
	dt = tplot/plotgap
	nplots = int(round(tmax/tplot))
	data = np.zeros((nplots+2,N))
	data[0,:] = v
	tdata = [t]
	for i in range(1,nplots+1):
		for n in range(1,plotgap+1):
			t = t+dt
			# print t
			# v_hat = fft(v)
			# w_hat = 1j*k* v_hat
			# w = np.real(ifft(w_hat)) 
			w = np.real(ifft(1j*k* fft(v)))
			vnew = vold - 2*dt*c*w
			vold = v
			v = vnew
		data[i+1,:] = v
		tdata.append(t)
	tdata = np.array(tdata)
	fig = plt.figure()
	#---- First subplot
	# ax = fig.add_subplot(121, projection='3d')
	ax = fig.gca(projection='3d')
	ax.view_init(elev=51., azim=-167)
	
	tv, xv = np.meshgrid((tdata).reshape((tdata.shape[0],)),(x).reshape((N,)),indexing='ij')
	surf = ax.plot_wireframe(tv, xv, data[:-1,:])
	
	ax.set_xlim(tdata[0], tdata[-1])
	ax.set_ylim(x[-1],x[0])
	ax.set_zlim(0., 3.)
	ax.set_xlabel('T')
	ax.set_ylabel('X')
	ax.set_zlabel('Z')
	plt.savefig('advection.pdf')
	plt.show()
	return



def bvp1_check():
	""" 
	Using scikits.bvp_solver to solve the bvp
	y'' = exp(-y) - .9*y, y(0) = y(2*pi) = 0
	y0 = y, y1 = y'
	y0' = y1, y1' = y'' = exp(-y0) - .9*y0
	"""
	from math import exp, pi
	lbc, rbc = 0., 0.
	
	def function1(x , y):
		return np.array([y[1] , exp(-y[0])-.9*y[0] ]) 
	
	
	def boundary_conditions(ya,yb):
		return (np.array([ya[0] - lbc]),  #evaluate the difference between the temperature of the hot stream on the
											 #left and the required boundary condition
				np.array([yb[0] - rbc]))#evaluate the difference between the temperature of the cold stream on the
											 #right and the required boundary condition
	
	problem = bvp_solver.ProblemDefinition(num_ODE = 2,
										  num_parameters = 0,
										  num_left_boundary_conditions = 1,
										  boundary_points = (0, 2.*pi),
										  function = function1,
										  boundary_conditions = boundary_conditions)
									
	solution = bvp_solver.solve(problem,
								solution_guess = (0.,
												  0.))
											
	A = np.linspace(0.,2.*pi, 200)
	T = solution(A)
	plt.plot(A, T[0,:],'-k',linewidth=2.0)
	plt.show()
	plt.clf()
	
	
	N = 150
	x = (2.*np.pi/N)*np.arange(1,N+1).reshape(N,1)
	print x.shape
	print solution(x)[0,:].shape
	plt.plot(x,solution(x)[0,:])
	plt.show()
	np.save('sol',solution(x)[0,:])
	return


def bvp1():
	N = 6
	x = (2.*np.pi/N)*np.arange(1,N+1).reshape(N,1)
	k = np.concatenate(( np.arange(0,N/2+1) ,
						 # np.array([0])	,
						 np.arange(-N/2+1,0,1)	))#.reshape(N,1)
	# print x.shape
	# print k.shape
	def g(v):
		# print v.shape
		v_hat = fft(v)
		# print v_hat.shape
		vpp_hat = (1j*k)**2*v_hat
		# print vpp_hat.shape
		vpp = np.real(ifft(vpp_hat))
		# print vpp.shape
		# import sys; sys.exit()
		
		w = vpp - np.exp(-v) + .9*v
		w[-1] = v[-1]-0.
		return w
	
	guess = 1.-np.cos(x)
	# guess = np.load('sol.npy')
	solution = root(g, guess)
	# print solution
	plt.plot(x,guess,'-r',linewidth=2.0)
	plt.plot(x,solution.x,'-k',linewidth=2.0)
	plt.show()
	return



def bvp2_check():
	""" 
	Using scikits.bvp_solver to solve the bvp
	y'' + y' + sin y = 0, y(0) = y(2*pi) = 0
	y0 = y, y1 = y'
	y0' = y1, y1' = y'' = -sin(y0) - y1
	"""
	from math import exp, pi, sin
	lbc, rbc = .1, .1
	
	def function1(x , y):
		return np.array([y[1] , -sin(y[0]) -y[1] ]) 
	
	
	def boundary_conditions(ya,yb):
		return (np.array([ya[0] - lbc]),  #evaluate the difference between the temperature of the hot stream on the
											 #left and the required boundary condition
				np.array([yb[0] - rbc]))#evaluate the difference between the temperature of the cold stream on the
											 #right and the required boundary condition
	
	problem = bvp_solver.ProblemDefinition(num_ODE = 2,
										  num_parameters = 0,
										  num_left_boundary_conditions = 1,
										  boundary_points = (0, 2.*pi),
										  function = function1,
										  boundary_conditions = boundary_conditions)
	
	guess = np.linspace(0.,2.*pi, 10)
	guess = np.array([.1-np.sin(2*guess),np.sin(2*guess)])
	# plt.plot(guess,np.sin(guess))
	# plt.show()
	
	solution = bvp_solver.solve(problem,
								solution_guess = guess)
	#										
	A = np.linspace(0.,2.*pi, 200)
	T = solution(A)
	plt.plot(A, T[0,:],'-k',linewidth=2.0)
	plt.show()
	plt.clf()
	
	
	N = 150
	x = (2.*np.pi/N)*np.arange(1,N+1).reshape(N,1)
	print x.shape
	print solution(x)[0,:].shape
	plt.plot(x,solution(x)[0,:])
	plt.show()
	np.save('sol',solution(x)[0,:])
	return


def bvp2():
	N = 150
	x = (2.*np.pi/N)*np.arange(1,N+1).reshape(N,1)
	k1 = np.concatenate(( np.arange(0,N/2) ,
						 np.array([0])	,
						 np.arange(-N/2+1,0,1)	))#.reshape(N,1)
	k2 = np.concatenate(( np.arange(0,N/2+1) ,
						 # np.array([0])	,
						 np.arange(-N/2+1,0,1)	))#.reshape(N,1)
	def g(v):
		v_hat = fft(v)
		vp_hat = (1j*k1)*v_hat
		vpp_hat = (1j*k2)**2*v_hat
		vpp = ifft(vpp_hat)
		vp = ifft(vp_hat)
		w = np.real(vpp	 + np.sin(v) + vp)
		w[-1] = v[-1]-.1
		return w
	
	# guess = 1.-np.cos(x)
	# guess = .1-np.sin(2*x)
	guess = np.load('sol.npy')
	solution = root(g, guess)
	print solution
	plt.plot(x,guess,'-r',linewidth=2.0)
	plt.plot(x,solution.x,'-k',linewidth=2.0)
	plt.show()
	return


def func0():
	nu, L = .1, 1.
	A, B, T = -L, L, 2.
	time_steps = 200
	delta_t = T/time_steps
	u_m,u_p = 5.,1.
	s, a = (u_m + u_p)/2., (u_m - u_p)/2.
	
	N = 200
	D,x = cheb(N)
	M = D.dot(D)
	M[0,:], M[-1,:] = 0., 0.
	M[0,0], M[-1,-1] = 1., 1.
	D2 = D[1:-1,:]
	
	bcs = np.zeros(N+1)
	bcs[0], bcs[-1] = 1., 5.
	def cap_F(u): 
		out = M.dot(u) - bcs
		out[1:-1] += (L/nu)*(s-u[1:-1])*D2.dot(u)
		return out
	
	guess = s - a*np.tanh(.6*(a/(2.*nu))*x)
	sol = root(cap_F, guess )
	
	if sol.success: 
		soln = s - a*np.tanh((a/(2.*nu))*(1./2)*(B+A + (B-A)*x))
		plt.plot((1./2)*(B+A + (B-A)*x),soln,'-k',linewidth=2.0,label='True solution')
		plt.plot((1./2)*(B+A + (B-A)*x),sol.x,'-r',linewidth=2.0,label='Numerical solution')
		
		plt.plot(x,guess,'*r',linewidth=2.0,label='Guess')
		plt.axis([-L,L,.5,5.5])
		plt.legend(loc='best')
		plt.show()
	return 



def Burgers():
	nu, L = .1, 2.
	u_m,u_p = 5.,1.
	s, a = (u_m + u_p)/2., (u_m - u_p)/2.
	
	def guess(x): 
		return s - a*np.tanh(a/(2.*nu)*x)
	
	N = 50
	D,x = cheb(N)
	M = D.dot(D)
	M[0,:], M[-1,:] = 0., 0.
	M[0,0], M[-1,-1] = 1., 1.
	D2 = D[1:-1,:]
	
	u_bcs = np.zeros(N+1)
	y_bcs = np.zeros(N+1)
	u_bcs[0], u_bcs[-1] = u_p, s
	y_bcs[0], y_bcs[-1] = u_m, s
	def cap_F(u): 
		out = np.zeros(2*(N+1))
		out1, out2 = out[:N+1], out[N+1:]
		u1, u2 = u[:N+1], u[N+1:]
		
		out1 += M.dot(u1) - u_bcs
		out1[1:-1] += (.5*L/nu)*(s - u1[1:-1])*D2.dot(u1)
		out2 += M.dot(u2) - y_bcs
		out2[1:-1] -= (.5*L/nu)*(s - u2[1:-1])*D2.dot(u2)
		return out
	
	array = np.empty(2*(N+1))
	array[:N+1] = guess(.4*L/2.*(x+1.))
	array[N+1:] = guess(-(.3)*L/2.*(x+1.))
	
	sol = root(cap_F, array )
	if sol.success:
		plt.plot(x,sol.x[:N+1],'-*k',linewidth=2.)
		plt.plot(x,sol.x[N+1:],'-*k',linewidth=2.)
		# plt.plot(x,array[:N+1],'-*r')
		# plt.plot(x,array[N+1:],'-*r')
		plt.axis([-1,1,u_p-.5, u_m+.5])
		plt.show()
		x_orig = L/2.*(x+1.)
		plt.plot(x_orig,sol.x[:N+1],'-ok',markersize=3.,linewidth=2.,label='Numerical solution')
		plt.plot(-x_orig,sol.x[N+1:],'-ok',markersize=3.,linewidth=2.)
		plt.plot(x_orig,array[:N+1],'-or',markersize=3.,linewidth=2.,label='Guess')
		plt.plot(-x_orig,array[N+1:],'-or',markersize=3.,linewidth=2.)
		plt.axis([-L,L,u_p-.5, u_m+.5])
		plt.legend(loc='best')
		plt.show()
	return 



def Solitons():
	# nu, L = .1, 2.
	# u_m,u_p = 5.,1.
	# s, a = (u_m + u_p)/2., (u_m - u_p)/2.
	L = 120. 
	s = .5
	
	def guess(x): 
		return (3.*s/2)*np.cosh( (math.sqrt(s/2.)/2.)*(x+3.5) )**(-2.)
	
	
	# x = np.linspace(-1,1,400)
	# plt.plot(x,guess(x),'-k')
	# plt.show()
	# import sys; sys.exit()
	
	# N = 50
	# D,x = cheb(N)
	# M = D.dot(D)
	# M[0,:], M[-1,:] = 0., 0.
	# M[0,0], M[-1,-1] = 1., 1.
	# D2 = D[1:-1,:]
	
	N = 50
	D1, x = cheb(N)
	D2 = D1.dot(D1)
	D3 = D2.dot(D1)
	D2[0,:], D2[-1,:] = 0., 0.
	D2[0,0], D2[-1,-1] = 1., 1.
	D1mod = D1[1:-1,:]
	
	u_bcs = np.zeros(N+1)
	y_bcs = np.zeros(N+1)
	u_bcs[0], u_bcs[-1] = 0., (3./3)*s
	y_bcs[0], y_bcs[-1] = 0., (3./3)*s
	def cap_F(u): 
		out = np.zeros(2*(N+1))
		out1, out2 = out[:N+1], out[N+1:]
		u1, u2 = u[:N+1], u[N+1:]
		
		out1 += D2.dot(u1) - u_bcs
		out1[1:-1] -= (s - u1[1:-1]/2.)*u1[1:-1]
		out2 += D2.dot(u2) - y_bcs
		out2[1:-1] -= (s - u2[1:-1]/2.)*u2[1:-1]
		return out
	
	array = np.empty(2*(N+1))
	array[:N+1] = guess(.4*L/2.*(x+1.))
	array[N+1:] = guess(-(.3)*L/2.*(x+1.))
	print np.min(array[N+1:]), np.min(array[:N+1])
	sol = root(cap_F, array )
	if sol.success:
		# print sol
		plt.plot(x,sol.x[:N+1],'-*k',linewidth=2.)
		plt.plot(x,sol.x[N+1:],'-*k',linewidth=2.)
		# plt.plot(x,array[:N+1],'-*r')
		# plt.plot(x,array[N+1:],'-*r')
		# plt.axis([-1,1,0.-.01, 2.*s])
		plt.show()
		x_orig = L/2.*(x+1.)
		plt.plot(x_orig,sol.x[:N+1],'-ok',markersize=3.,linewidth=2.,label='Numerical solution')
		plt.plot(-x_orig,sol.x[N+1:],'-ok',markersize=3.,linewidth=2.)
		plt.plot(x_orig,array[:N+1],'-or',markersize=3.,linewidth=2.,label='Guess')
		plt.plot(-x_orig,array[N+1:],'-or',markersize=3.,linewidth=2.)
		# plt.axis([-L,L,0.-.01, 2.*s])
		plt.legend(loc='best')
		plt.show()
	return 




def tref_solitons():
	# p27.m - Solve KdV eq. u_t + uu_x + u_xxx = 0 on [-pi,pi] by
	#		 FFT with integrating factor v = exp(-ik^3t)*u-hat.
	
	# Set up grid and two-soliton initial data:
	N = 256
	dt = .4*N**(-2.)
	x = (2.*np.pi/N)*np.arange(-N/2,N/2).reshape(N,1)
	A, B = 25., 16.
	u = 3.*A**2.*np.cosh(.5*(A*(x+2.)))**(-2.) + 3*B**2.*np.cosh(.5*(B*(x+1.)))**(-2.);
	u = u.reshape(N,1) 
	v = fft(u,axis=0)
	# k = [0:N/2-1 0 -N/2+1:-1]'
	k = np.concatenate(( np.arange(0,N/2) ,
						 np.array([0])	,
						 np.arange(-N/2+1,0,1)	)).reshape(N,1)
	ik3 = 1j*k**3.
	print "u = ", u
	print "v = ", v
	# print "k = ", k # This is correct
	# Solve PDE and plot results:
	tmax = 0.006;
	nplt = int(np.floor((tmax/45.)/dt))
	nmax = int(round(tmax/dt))
	
	print "nplt = ", nplt
	print "nmax = ", nmax
	udata, tdata = u, np.array(0.).reshape(1,1)
	for n in range(1,nmax+1): #= 1:nmax
		t = n*dt
		g = -.5j*dt*k
		E = np.exp(dt*ik3/2.)
		E2 = E**2.
		a = g*fft(np.real( ifft(	   v	,axis=0) )**2.,axis=0)
		b = g*fft(np.real( ifft(E*(v+a/2.),axis=0) )**2.,axis=0)	 # 4th-order
		c = g*fft(np.real( ifft(E*v + b/2.,axis=0) )**2.,axis=0)	 # Runge-Kutta
		d = g*fft(np.real( ifft(E2*v+E*c,axis=0) )**2.,axis=0)
		v = E2*v + (E2*a + 2*E*(b+c) + d)/6.
		if mod(n,nplt) == 0:
			u = np.real(ifft(v,axis=0))
			# print n
			# print u
			udata = np.concatenate((udata,np.nan_to_num(u)),axis=1)
			tdata = np.concatenate((tdata,np.array(t).reshape(1,1)),axis=1)
	
	fig = plt.figure()# figsize=plt.figaspect(0.5))
	#---- First subplot
	# ax = fig.add_subplot(121, projection='3d')
	ax = fig.gca(projection='3d')
	# ax.view_init(elev=40., azim=70)
	ax.view_init(elev=20., azim=30)
	tv, xv = np.meshgrid(tdata.reshape((tdata.shape[1],)),x.reshape((N,)),indexing='ij')
	print tv.shape
	print xv.shape
	print udata.shape
	surf = ax.plot_surface(tv, xv, udata.T, rstride=5, cstride=5, cmap=cm.coolwarm,
		linewidth=0, antialiased=False)
	tdata = tdata[0]
	print tdata
	ax.set_xlim(tdata[0], tdata[-1])
	ax.set_ylim(x[0], x[-1])
	ax.set_zlim(0., 10000.)
	ax.set_xlabel('T')
	ax.set_ylabel('X')
	ax.set_zlabel('Z')
	plt.show()
	  # waterfall(x,tdata,udata'), colormap(1e-6*[1 1 1]); view(-20,25)
	  # xlabel x, ylabel t, axis([-pi pi 0 tmax 0 2000]), grid off
	  # set(gca,'ztick',[0 2000]), close(h), pbaspect([1 1 .13])
	return 
	



if __name__ == "__main__":
	# deriv1()
	# advection()
	# bvp1_check()
	# bvp1()
	# bvp2_check()
	# bvp2()
	
	# func0()
	# Burgers()
	# Solitons()
	tref_solitons()