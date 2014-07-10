from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
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
		return -m**2.*2.*(x-1./2.)*f(x)
	
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



# ffmpeg -r 9 -i wave%03d.png -r 25 -qscale 1 out.mpg
# ffmpeg -r 17 -i wave%03d.png -r 25 -qscale 1 out5.mpg
# example1()
example2()
# example3()
# example4()
# example5()
