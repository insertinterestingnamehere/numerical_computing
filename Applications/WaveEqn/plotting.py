from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

from solution import wave_1d, math_animation



def example1():
	L,     T  =  1., 2
	N_x, N_t  =  200, 440
	def f(x):
		return np.sin(2*np.pi*x) + 4.*np.sin(np.pi*x) + .2*np.sin(4*np.pi*x)
	
	def g(x):
		return np.zeros(x.shape)
	
	view = [0-.1,L+.1,-5,5]
	Data = wave_1d(f,g,L,N_x,T,N_t,view)
	stride = 4
	Data = Data[0], Data[1][::stride,:]
	time_steps, wait = int(N_t/stride), 2
	
	math_animation(Data,time_steps,view,wait)
	return 


def example2():
	L,     T  =  1., 2
	N_x, N_t  =  200, 440
	m, u_0 = 20, .2
	def f(x):
		return u_0*np.exp(-m**2.*(x-1./2.)**2.)
	
	def g(x):
		return m**2.*2.*(x-1./2.)*f(x)
	
	view = [0-.1,L+.1,-.5,.5]
	Data = wave_1d(f,g,L,N_x,T,N_t,view)
	stride = 1
	Data = Data[0], Data[1][::stride,:]
	time_steps, wait = int(N_t/stride), 4
	
	math_animation(Data,time_steps,view,wait)
	return


def example3():
	L,     T  =  1., 2
	N_x, N_t  =  200, 440
	
	m, u_0 = 20, 1/3.
	def f(x):
		return u_0*np.exp(-m**2.*(x-1./2.)**2.)
	
	def g(x):
		return np.zeros(x.shape)
	
	view = [-.1,L+.1,-.5,.5]
	Data = wave_1d(f,g,L,N_x,T,N_t,view)
	stride = 2
	Data = Data[0], Data[1][::stride,:]
	time_steps, wait = int(N_t/stride), 80
	
	math_animation(Data,time_steps,view,wait)
	return


def example4():
	L,     T  =  1., 2
	N_x, N_t  =  200, 440
	def f(x):
		y = np.zeros_like(x)
		y[np.where( (5.*L/11. < x) & (x < 6*L/11. ) ) ] = 1./3
		return y
	
	def g(x):
		return np.zeros(x.shape)
	
	view = [-.1,L+.1,-.5,.5]
	Data = wave_1d(f,g,L,N_x,T,N_t,view)
	stride = 2
	Data = Data[0], Data[1][::stride,:]
	time_steps, wait = int(N_t/stride), 80
	
	math_animation(Data,time_steps,view,wait)
	return


def example5():
	L,     T  =  1., 2
	N_x, N_t  =  200, 440
	def f(x):
		return 4.*np.sin(4*np.pi*x)
	
	def g(x):
		return np.zeros(x.shape)
	
	view = [0-.1,L+.1,-5,5]
	Data = wave_1d(f,g,L,N_x,T,N_t,view)
	stride = 2
	Data = Data[0], Data[1][::stride,:]
	time_steps, wait = int(N_t/stride), 200
	
	math_animation(Data,time_steps,view,wait)
	return 


def Burgers():
	
	def bump(x): 
		sigma, mu = 1., 0.
		gaussian = (1./(sigma*np.sqrt(2*np.pi) ) )*np.exp((-1./2.)*((x-mu)/sigma)**2.)
		return gaussian*(3.5*np.sin(3*x)+ 3.5)
	
	def pde_matrix_func(Y2,Y1):
		out = np.zeros(len(Y1))
		
		c1 = (theta*delta_t)/(2.*delta_x)
		c2 = (theta*delta_t)/(delta_x**2.)
		c3 = ((1.-theta)*delta_t)/(2.*delta_x)
		c4 = ((1.-theta)*delta_t)/(delta_x**2.)
		
		out[1:-1] = ( Y2[1:-1] + c1*(Y2[1:-1] - s)*(Y2[2:] - Y2[:-2]) -
						c2*(Y2[2:] - 2.*Y2[1:-1] + Y2[:-2])- 
						Y1[1:-1] + c3*(Y1[1:-1] - s)*(Y1[2:] - Y1[:-2]) -
						c4*(Y1[2:] - 2.*Y1[1:-1] + Y1[:-2])       
					)
		
		
		out[0] = Y2[0]-Y1[0]		
		out[-1] = Y2[-1]-Y1[-1]	
		# for j in range(len(Y1)-1,len(Y1)): 	out[j] = Y2[j]-Y1[j]	
		return out
	
	
	L, T = 20., 1.
	u_m,u_p = 5.,1.
	s, a = (u_m + u_p)/2., (u_m - u_p)/2.
	
	n, time_steps = 150, 350
	delta_x, delta_t = 2.*L/n, T/(n)
	x = np.linspace(-L,L,n)
	wave, perturb = s - a*np.tanh((a/2.)*x), bump(x)
	theta = 1/2.
	
	U = np.zeros( (time_steps, n) )
	U[0,:] = wave+perturb
	U[0,0], U[0,-1] = u_m, u_p
	
	for j in xrange(1, time_steps):
		U[j,:] = fsolve(lambda X,Y= U[j-1,:]: pde_matrix_func(X,Y), U[j-1,:] )
		print j
	
	view = [-L,L,u_p-1,u_m+1]
	# Data = wave_1d(f,g,L,N_x,T,N_t,view)
	stride = 1
	Data = x, U[::stride,:], wave
	time_steps, wait = int(time_steps/stride), 30
	
	math_animation(Data,time_steps,view,wait)
	return


Burgers()
# To create the movie, run ffmpeg -r 12 -i burgers%03d.png -r 25 -qscale 1 movie_burgers.mpg
# from the command line
# ffmpeg -r 9 -i wave%03d.png -r 25 -qscale 1 out.mpg
# ffmpeg -r 17 -i wave%03d.png -r 25 -qscale 1 out5.mpg
# example1()
# example2()
# example3()
# example4()
example5()
