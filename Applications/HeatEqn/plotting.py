# Solving u_t = nu u_{xx}, t \in [0,T], x \in [-L, L]
import numpy as np
from solution import heatexplicit, math_animation
import matplotlib.pyplot as plt
from matplotlib import cm

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
	Domain = 10.
	Data = heatexplicit(J=320,nu=1.,L=Domain,T=1.,flag3d="animation")
	
	Data = Data[0], Data[1].T[0:-1,:]
	time_steps, view = Data[1].shape[0], [-Domain, Domain,-.1, 1.5]
	
	math_animation(Data,time_steps,view)
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
	# animate_heat1d()


