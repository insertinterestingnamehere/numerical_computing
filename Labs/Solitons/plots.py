from __future__ import division
import numpy as np					
from scipy.fftpack import fft, ifft		
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm			 

from math import sqrt, pi

def initialize_all(y0, t0, t1, n):
	""" An initialization routine for the different ODE solving
	methods in the lab. This initializes Y, T, and h. """
	
	if isinstance(y0, np.ndarray):
		Y = np.empty((n, y0.size),dtype=complex).squeeze()
	else:
		Y = np.empty(n,dtype=complex)
	Y[0] = y0
	T = np.linspace(t0, t1, n)
	h = float(t1 - t0) / (n - 1)
	return Y, T, h



def RK4(f, y0, t0, t1, n):
	""" Use the RK4 method to compute an approximate solution
	to the ODE y' = f(t, y) at n equispaced parameter values from t0 to t
	with initial conditions y(t0) = y0.
	
	'y0' is assumed to be either a constant or a one-dimensional numpy array.
	't0' and 't1' are assumed to be constants.
	'f' is assumed to accept two arguments.
	The first is a constant giving the current value of t.
	The second is a one-dimensional numpy array of the same size as y.
	
	This function returns an array Y of shape (n,) if
	y is a constant or an array of size 1.
	It returns an array of shape (n, y.size) otherwise.
	In either case, Y[i] is the approximate value of y at
	the i'th value of np.linspace(t0, t, n).
	"""
	Y, T, h = initialize_all(y0, t0, t1, n)
	for i in xrange(1, n):
		K1 = f(T[i-1], Y[i-1])
		# print "Y[i-1].shape = ", Y[i-1].shape
		tplus = (T[i] + T[i-1]) * .5
		K2 = f(tplus, Y[i-1] + .5 * h * K1)
		K3 = f(tplus, Y[i-1] + .5 * h * K2)
		K4 = f(T[i], Y[i-1] + h * K3)
		# print "K1 + 2 * K2 + 2 * K3 + K4.shape = ", (K1 + 2 * K2 + 2 * K3 + K4).shape
		Y[i] = Y[i-1] + (h / 6.) * (K1 + 2 * K2 + 2 * K3 + K4)
	return T, Y



def plot_soliton():
	N = 256
	# grid = np.linspace(0,2.*pi,	N)
	# s1, a1 = 25.**2., 2.
	# y1 = 3*s1*np.cosh(sqrt(s1)/2.*(grid-a1))**(-2.)
	# s2, a2 = 16.**2., 1.
	# y2 = 3*s2*np.cosh(sqrt(s2)/2.*(grid-a2))**(-2.)	 
	# plt.plot(grid,y1,'-k',linewidth=2.)		
	# plt.plot(grid,y2,'-b',linewidth=2.)		
	# plt.show()
	
	def unScaled():
		# Set up grid and two-soliton initial data:
		x = (2.*np.pi/N)*np.arange(-N/2,N/2).reshape(N,1)
		A, B = 25., 16.
		A_shift, B_shift = 2., 1.
		y0 = (3.*A**2.*np.cosh(.5*(A*(x+2.)))**(-2.) + 3*B**2.*np.cosh(.5*(B*(x+1.)))**(-2.)).reshape(N,)
		k = np.concatenate(( np.arange(0,N/2) ,
							 np.array([0])	,
							 np.arange(-N/2+1,0,1)	)).reshape(N,)
		ik3 = 1j*k**3.
		
		def F_unscaled(t,u):
			out = -.5*1j*k*fft(ifft(u,axis=0)**2.,axis=0)  + ik3* u			
			return out
		
		
		tmax = .006
		dt = .01*N**(-2.)
		nmax = int(round(tmax/dt))
		nplt = int(np.floor((tmax/25.)/dt))
		y0 = fft(y0,axis=0)
		T,Y = RK4(F_unscaled, y0, t0=0, t1=tmax, n=nmax)
		
		udata, tdata = np.real(ifft(y0,axis=0)).reshape(N,1), np.array(0.).reshape(1,1)
		for n in range(1,nmax+1):
			if np.mod(n,nplt) == 0:
				t = n*dt
				u = np.real( ifft(Y[n], axis=0) ).reshape(N,1)
				udata = np.concatenate((udata,np.nan_to_num(u)),axis=1)
				tdata = np.concatenate((tdata,np.array(t).reshape(1,1)),axis=1)
		
		return x, tdata, udata
	
	
	
	
	def Scaled():
		# Set up grid and two-soliton initial data:
		x = (2.*np.pi/N)*np.arange(-N/2,N/2).reshape(N,1)
		A, B = 25., 16.
		A_shift, B_shift = 2., 1.
		y0 = (3.*A**2.*np.cosh(.5*(A*(x+2.)))**(-2.) + 3*B**2.*np.cosh(.5*(B*(x+1.)))**(-2.)).reshape(N,)
		k = np.concatenate(( np.arange(0,N/2) ,
							 np.array([0])	,
							 np.arange(-N/2+1,0,1)	)).reshape(N,)
		ik3 = 1j*k**3.
		
		def F_scaled(t,U):
			E = np.exp(-ik3*t)
			E_recip = E**(-1.)
			out = -.5*1j*E*k*fft(ifft(E_recip*U,axis=0)**2.,axis=0)				 
			return out
		
		
		tmax = .006
		dt = .2*N**(-2.)
		nmax = int(round(tmax/dt))
		nplt = int(np.floor((tmax/25.)/dt))
		y0 = fft(y0,axis=0)
		T,Y = RK4(F_scaled, y0, t0=0, t1=tmax, n=nmax)
		
		udata, tdata = np.real(ifft(y0,axis=0)).reshape(N,1), np.array(0.).reshape(1,1)
		for n in range(1,nmax+1):
			if np.mod(n,nplt) == 0:
				t = n*dt
				u = np.real(ifft(np.exp(ik3*t)*(Y[n]),axis=0)).reshape(N,1)
				udata = np.concatenate((udata,np.nan_to_num(u)),axis=1)
				tdata = np.concatenate((tdata,np.array(t).reshape(1,1)),axis=1)
		
		return x, tdata, udata
	
	
		
	# x, tdata, udata = Scaled()
	x, tdata, udata = unScaled()
	# import sys; sys.exit()
	fig = plt.figure()# figsize=plt.figaspect(0.5))
	#---- First subplot
	# ax = fig.add_subplot(121, projection='3d')
	ax = fig.gca(projection='3d')
	ax.view_init(elev=45., azim=150)
	tv, xv = np.meshgrid(tdata,x,indexing='ij')
	surf = ax.plot_surface(tv, xv, udata.T, rstride=1, cstride=1, cmap=cm.coolwarm,
		linewidth=0, antialiased=False)
	
	tdata = tdata[0]
	ax.set_xlim(tdata[0], tdata[-1])
	ax.set_ylim(-pi, pi)
	ax.invert_yaxis()
	ax.set_zlim(0., 4000.)
	ax.set_xlabel('T'); ax.set_ylabel('X'); ax.set_zlabel('Z')
	# plt.savefig('interacting_solitons.png',dpi=100)
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
	return


	






if __name__ == "__main__": 
	plot_soliton()
	
	
	
	
