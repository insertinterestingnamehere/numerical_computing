from __future__ import division
import numpy as np
from matplotlib import animation
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.artist import setp
plt.switch_backend('tkagg')
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.sparse import spdiags, coo_matrix, bmat, identity
from scipy.sparse.linalg import spsolve


def heatexplicit(init_conditions,x_subintervals,t_subintervals,x_interval=[-10,10],T=1.,flag3d="off",nu=1.):
	'''
	Parameters
	nu: diffusive constant
	L, T: Solve on the Cartesian rectangle (x,t) in x_interval x [0,T]
	x_subintervals: Number of subintervals in spatial dimension
	t_subintervals: Number of subintervals in time dimension
	CFL condition: nu delta_t/delta_x**2 <= 1/2. Where delta_t = T/t_subintervals, 
	delta_x = (b-a)/x_subintervals, 
	a, b = x_interval[0], x_interval[1]
	In terms of our constants, this means that t_subintervals >=(2.*nu)*(T*x_subintervals**2/(b-a)**2.) or 
	equivalently that (nu/4.)*(T/L**2.)*(x_subintervals**2./t_subintervals) <= 1/2 
	'''
	a, b = x_interval[0], x_interval[1]
	
	delta_x, delta_t = (b-a)/x_subintervals, T/t_subintervals
	if nu*delta_t/delta_x**2 > 1/2: 
		print "The CFL condition is not satisfied"
		print "Must have nu*delta_t/delta_x**2 <= 1/2, i.e."+str(nu*delta_t/delta_x**2) + "<= 1/2"
		print "Recommend t_subintervals = "+str(t_subintervals)+">" +str((2.*nu)*(T*x_subintervals**2/(b-a)**2.))
		
	K = nu*delta_t/delta_x**2.
	# print str(J)+" subintervals of Space domain.\n"
	# print str(int(N))+" subintervals of Time domain." 
	
	D0,D1,diags = (1.-2.*K)*np.ones((1,(x_subintervals-1))), K*np.ones((1,(x_subintervals-1))), np.array([0,-1,1])
	data = np.concatenate((D0,D1,D1),axis=0) # This stacks up rows
	A=spdiags(data,diags,(x_subintervals-1),(x_subintervals-1)).asformat('csr') 
	
	U = np.zeros((x_subintervals+1,t_subintervals+1)) #[:,0:-1]
	U[:,0] = init_conditions #np.maximum(1.-np.linspace(a,b,J+1)**2,0.)	# Initial Conditions
	for j in range(0,int(t_subintervals)+1):
		if j>0: U[1:-1,j] =  A*U[1:-1,j-1]
	
	return np.linspace(a,b,x_subintervals+1), U


def heat_Crank_Nicolson(init_conditions,x_subintervals,t_subintervals,x_interval=[-10,10],T=1.,flag3d="off",nu=1.):
	'''
	Parameters
	nu: diffusive constant
	L, T: Solve on the Cartesian rectangle (x,t) in x_interval x [0,T]
	x_subintervals: Number of subintervals in spatial dimension
	t_subintervals: Number of subintervals in time dimension
	a, b = x_interval[0], x_interval[1]
	'''
	a, b = x_interval[0], x_interval[1]
	delta_x, delta_t = (b-a)/x_subintervals, T/t_subintervals
		
	K = .5*nu*delta_t/delta_x**2.
		
	D0,D1,diags = (1-2.*K)*np.ones((1,(x_subintervals-1))), K*np.ones((1,(x_subintervals-1))), np.array([0,-1,1])
	data = np.concatenate((D0,D1,D1),axis=0) # This stacks up rows
	A=spdiags(data,diags,(x_subintervals-1),(x_subintervals-1)).asformat('csr') 
	# print K
	# print A.todense()
	D0,D1,diags = (1.+2.*K)*np.ones((1,(x_subintervals-1))), -K*np.ones((1,(x_subintervals-1))), np.array([0,-1,1])
	data = np.concatenate((D0,D1,D1),axis=0) # This stacks up rows
	B=spdiags(data,diags,(x_subintervals-1),(x_subintervals-1)).asformat('csr')
	
	U = np.zeros((x_subintervals+1,t_subintervals+1))
	U[:,0] = init_conditions 
	for j in range(0,int(t_subintervals)+1):
		if j>0: U[1:-1,j] =  spsolve(B,A*U[1:-1,j-1] )
	return np.linspace(a,b,x_subintervals+1), U


a = 0
def math_animation(Data,time_steps,view,interval_length):
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
			setp(line, linewidth=2, color='k')
			# line = Line2D(X, Array[:,i], color='red', linewidth=2)
		else:
			line.set_data(X, Array[-1,:])
		a+=1
		
		return line
	
	# call the animator.  blit=True means only re-draw the parts that have changed.
	anim = animation.FuncAnimation(fig, animate_function, init_func=initialize_background,
                               frames=time_steps, interval=interval_length)#, blit=True)
	# frames must be a generator, an iterable, or a number of frames
	# Draws a new frame every interval milliseconds.
	
	plt.show()
	return 


# Solving u_t = nu u_{xx}, t \in [0,T], x \in [-L, L]
# import numpy as np
# from solution import heatexplicit, math_animation
# import matplotlib.pyplot as plt
# from matplotlib import cm

def plot3d():
	T, L, J, nu = 1., 10., 320, 1.
	flag3d = "3d_plot"
	
	X,U = heatexplicit(J,nu,L,T,flag3d)
	N = U.shape[1]-1
	# #Produce 3D plot of solution
	xv,tv = np.meshgrid(np.linspace(-L,L,J/4+1), np.linspace(0,T,N/8+1))
	Z = U[::4,::8].T
	
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	surf = ax.plot_surface(tv, xv, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
	    								linewidth=0, antialiased=False)
	ax.set_xlabel('Y'); ax.set_ylabel('X'); ax.set_zlabel('Z')
	plt.show()
	return 


def animate_heat1d():
	x_subintervals, t_subintervals = 20,1600
	x_interval,t_final = [0,1],2
	init_conditions = np.sin(2.*np.pi*(np.linspace(0,1,x_subintervals+1)))
	Data = heatexplicit(init_conditions,x_subintervals,t_subintervals,x_interval,T=t_final,flag3d="on",nu=1.)
	
	Data = Data[0], Data[1].T[0:-1,:]
	time_steps, view = Data[1].shape[0], [-.1, 1.1,-1.1, 1.1]
	interval=40
	math_animation(Data,time_steps,view,interval)
	return 


def plot_error():
	# Compare solutions for various discretizations
	X, U, L = [], [], 10.
	List, LineStyles = [40,80,160,320,640,1280,2560], ['--k','-.k',':k','-k','-k','-k']
	for points in List:
		x,u = heatexplicit(J=points,nu=1.,L=10.,T=1.,flag3d="off")
		X.append(x); 
		U.append(u[:,-1])
		del x, u
		print points
	# # Plot of solutions for varying discretizations
	# for j in range(0,len(LineStyles)): plt.plot(X[j],U[j],LineStyles[j],linewidth=1)
	# plt.axis([-10,10,-.1,.4])
	# plt.show()
	# plt.clf()
	h = [2.*L/item for item in List[:-1]] 
	
	def graph(X,U,List,h):
		# # Error Chart 1: Plot approximate error for each solution
		# for j in range(0,len(List)-1): plt.plot(X[j],abs(U[-1][::2**(6-j)]-U[j]),'-k',linewidth=1.2)
		for j in range(0,len(List)-1): plt.plot(X[j],abs(U[-1][::List[-1]/List[j]]-U[j]),'-k',linewidth=1.2)
		plt.savefig("ApproximateError.pdf")
		plt.clf()
		
		# # Error Chart 2: Create a log-log plot of the max error of each solution
		
		# MaxError = [max(abs(U[-1][::2**(6-j)]-U[j] )) for j in range(0,len(List)-1)]
		MaxError = [max(abs(U[-1][::List[-1]/List[j]]-U[j] )) for j in range(0,len(List)-1)]
		plt.loglog(h,MaxError,'-k',h,MaxError,'ko')
		plt.ylabel("Max Error")
		plt.xlabel('$\Delta x$')
		h, MaxError = np.log(h)/np.log(10),np.log(MaxError)/np.log(10)
		print '\n'*2, "Approximate order of accuracy = ", (MaxError[-1]-MaxError[0])/(h[-1]-h[0])
		plt.savefig("MaximumError.pdf")
		# plt.show()
		return 
	
	
	graph(X,U,List,h)
	
	return



if __name__=="__main__":
	
	# plot3d()
	# plot_error()
	animate_heat1d()









