from __future__ import division
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.artist import setp
plt.switch_backend('tkagg')



def wave_1d(f,g,L,N_x,T,N_t,view):
	# Solves the wave equation u_{tt} = cu_{xx} where
	# Domain = [0,L], Time interval = [0,T],
	# and N_x and N_t represent the subintervals 
	# in the x and t dimensions
	delta_x, delta_t = L/N_x, T/N_t
	c = 1.
	lmbda = c*delta_t/delta_x
	U, x_grid = np.zeros((N_t+1, N_x+1)), np.linspace(0,L,N_x+1)
	f_grid, g_grid = f(x_grid), g(x_grid)
		
	U[1,1:-1] = f_grid[1:-1]
	U[0,1:-1] = ( -delta_t*g_grid[1:-1]  + U[1,1:-1] + 
					(1./2.)*(c*delta_t/delta_x)**2.*(U[1,2:] -2.*U[1,1:-1] + U[1,:-2])
					)
	
	for m in xrange(1,N_t): 
		U[m+1,1:-1] = ( (2.*U[m,1:-1]-U[m-1,1:-1]) + 
						(c*delta_t/delta_x)**2. * (U[m,2:] -2.*U[m,1:-1] + U[m,:-2]) 
						)
	
	# for m in xrange(1,N_t):
	# 	plt.plot(x_grid, U[m,:], '-r')
	# 	plt.axis(view)
	# 	filestring = '000%i'%m
	# 	plt.savefig('wave_1d_png/wave'+filestring[-3::])
	# 	plt.clf(); plt.close()
	# 	print m
		
	return x_grid, U



a = 0
def math_animation(Data,time_steps,view,wait):
	X,Array = Data
	
	fig = plt.figure()
	ax = plt.axes(xlim=tuple(view[0:2]), ylim=tuple(view[2:]) )
	line, = ax.plot([], [], lw=2)
	
	def initialize_background():
		line.set_data([], [])
		return line,
	
	def animate_function(i):
		global a
		if a<time_steps:
			line.set_data(X, Array[i,:])
			setp(line, linewidth=2, color='r')
			# line = Line2D(X, Array[:,i], color='red', linewidth=2)
		else:
			line.set_data(X, Array[-1,:])
		a+=1
		
		return line
	
	
	
	# call the animator.  blit=True means only re-draw the parts that have changed.
	anim = animation.FuncAnimation(fig, animate_function, init_func=initialize_background,
                               frames=time_steps, interval=wait)#, blit=True)
	# frames must be a generator, an iterable, or a number of frames
	# Draws a new frame every interval milliseconds.
	
	plt.show()
	return 
