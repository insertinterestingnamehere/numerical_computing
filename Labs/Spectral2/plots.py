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
	nplt = int(np.floor((tmax/25.)/dt))
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
	



if __name__ == "__main__":
	# func0()
	# Burgers()
	# Solitons()
	tref_solitons()