from __future__ import division
import numpy as np
from matplotlib import animation, rcParams
from matplotlib import pyplot as plt
from matplotlib.artist import setp
plt.switch_backend('tkagg')
rcParams['figure.figsize'] = 12, 8.5
from matplotlib.pyplot import Line2D
import time






a = 0
def math_animation(Data,time_steps,view,wait):
	X,Array,Constant = Data
	fig = plt.figure()
	ax = plt.axes(xlim=tuple(view[0:2]), ylim=tuple(view[2:]) )
	line, = ax.plot([], [], lw=2.6,c='k')
	lines = [line]
	lines.append(ax.plot([], [], lw=2.6,c='r')[0])
	def initialize_background():
		if Constant==None: 
			lines[0].set_data([], [])
		else: 
			lines[0].set_data(X, Constant)
			# line.set_data([], [])
			# line += ax.plot(X, Constant, '-', c='k')
			
		return lines
	
	def animate_function(i):
		global a
		if i==0:
			time.sleep(.3)
		if a<time_steps:
			lines[1].set_data(X, Array[i,:])
			setp(lines[1], linewidth=2.6, color='k')
		else:
			lines[1].set_data(X, Array[-1,:])
			
		a+=1
		
		return lines
	
	
	
	# call the animator.  blit=True means only re-draw the parts that have changed.
	anim = animation.FuncAnimation(fig, animate_function, init_func=initialize_background,
                               frames=time_steps, interval=wait)#, blit=True)
	# frames must be a generator, an iterable, or a number of frames
	# Draws a new frame every interval milliseconds.
	
	plt.show()
	return 

