from __future__ import division
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.artist import setp
plt.switch_backend('tkagg')

from scipy.sparse import spdiags, coo_matrix, bmat, identity
from scipy.sparse.linalg import spsolve
from mpl_toolkits.mplot3d.axes3d import Axes3D

def heatexplicit(J=320,nu=1.,L=10.,T=1.,flag3d="off"):
	'''
	Parameters
	nu: diffusive constant
	L, T: Solve on the Cartesian rectangle (x,t) in [-L,L] x [0,T]
	J: Number of subintervals of [-L, L]
	N: Number of subintervals of [0, T]
	CFL condition: nu delta_t/delta_x**2 <= 1/2. (delta_t = 2L/J, delta_x = T/N )
	In terms of our constants, this means that N >=(nu/2.)*(T/L**2.)*J**2 or 
	equivalently that (nu/4.)*(T/L**2.)*(J**2./N) <= 1/2 
	'''
	N = (5./8.)*nu*(T/L**2.)*J**2.  
	delta_x, delta_t = 2.*L/J, T/N
	K = nu*delta_t/delta_x**2.
	# print str(J)+" subintervals of Space domain [-"+str(L)+"," +str(L)+"].\n"
	# print str(int(N))+" subintervals of Time domain [0, "+str(T)+"]." 
	
	D0,D1,diags = (1.-2.*K)*np.ones((1,(J-1))), K*np.ones((1,(J-1))), np.array([0,-1,1])
	data = np.concatenate((D0,D1,D1),axis=0) # This stacks up rows
	A=spdiags(data,diags,(J-1),(J-1)).asformat('csr') 

	U = np.zeros((J+1,N+1))[:,0:-1]
	U[:,0] = np.maximum(1.-np.linspace(-L,L,J+1)**2,0.)	# Initial Conditions
	for j in range(0,int(N)):
		if j>0: U[1:-1,j] =  A*U[1:-1,j-1]
	
	return np.linspace(-L,L,J+1), U






a = 0
def math_animation(Data,time_steps,view):
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
                               frames=time_steps, interval=20)#, blit=True)
	# frames must be a generator, an iterable, or a number of frames
	# Draws a new frame every interval milliseconds.
	
	plt.show()
	return 
