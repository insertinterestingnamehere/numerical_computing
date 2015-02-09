import numpy as np
from scipy.misc import derivative
from scipy.optimize import minimize, root
import matplotlib.pyplot as plt


def example_text():
	def f(x):
		out = -np.ones(x.shape)
		m = np.where(x<.5)
		out[m] = -6*x[m]**2. + 3.*x[m] - 1.
		return out
	
	
	def u(x):
		return (x+1./4)**2. + 1./4
	
	
	def int_f(x):
		out = np.zeros(x.shape)
		m = np.where(x<.5) # antiderivative here is 
		# -2x**3 + (3./2)*x**2 - x
		out[m] += -2*x[m]**3. + (3./2)*x[m]**2. - x[m]
		m = np.where(x>=.5) # antiderivative here is 
		# -x
		out[m] = -2*(1./2)**3. + (3./2)*(1./2)**2. - (1./2)
		out[m] += -1.*(x[m]-1./2)
		return out
	
	
	def plot_example_data():
		x = np.linspace(0,1,200)
		plt.plot(x,u(x),'-k',linewidth=2.,label='u')
		plt.plot(x,derivative(u,x,dx=1e-6),'-r',linewidth=2.,label="u'")
		plt.plot(x,2*x + 1./2,'*r',label="u'")
		
		plt.plot(x,a(x),'-b',linewidth=2.,label='a')
		plt.plot(x,(1./4)*(3-x),'*b',label='guess for a')
		plt.plot(x,-f(x),'-g',linewidth=2.,label='-f')
		
		plt.plot(x,int_f(x),'-g',linewidth=2.,label='int(f)')
		
		plt.legend(loc='best')
		plt.axis([0,1,-2,2])
		plt.show()
		plt.clf()
		return
	
	
	x = np.linspace(0,1,11)
	F, u_p = int_f(x), derivative(u,x,dx=1e-6)
	
	def least_squares(c):
		out = ( (3./8 - F)/c - u_p )**2.
		return np.sum(out)
	
	guess = (1./4)*(3-x)
	sol = minimize(least_squares,guess)
	
	plt.plot(x,sol.x,'-ob',linewidth=2)
	plt.show()
	# plt.clf()
	return 





def example():
	def f(x):
		out = -np.ones(x.shape)
		m = np.where(x<.5)
		out[m] = -6*x[m]**2. + 3.*x[m] - 1.
		return out
	
	
	def a(x):
		out = (1./2)*np.ones(x.shape)
		m = np.where(x<.5)
		out[m] = x[m]**2. - x[m] + 3./4
		return out
	
	
	def u(x):
		return (x+1./4)**2. + 1./4
	
	
	def int_f(x):
		out = np.zeros(x.shape)
		m = np.where(x<.5) # antiderivative here is 
		# -2x**3 + (3./2)*x**2 - x
		out[m] += -2*x[m]**3. + (3./2)*x[m]**2. - x[m]
		m = np.where(x>=.5) # antiderivative here is 
		# -x
		out[m] = -2*(1./2)**3. + (3./2)*(1./2)**2. - (1./2)
		out[m] += -1.*(x[m]-1./2)
		return out
	
	
	def plot_example_data():
		x = np.linspace(0,1,200)
		plt.plot(x,u(x),'-k',linewidth=2.,label='u')
		plt.plot(x,derivative(u,x,dx=1e-6),'-r',linewidth=2.,label="u'")
		plt.plot(x,2*x + 1./2,'*r',label="u'")
		
		plt.plot(x,a(x),'-b',linewidth=2.,label='a')
		plt.plot(x,(1./4)*(3-x),'*b',label='guess for a')
		plt.plot(x,-f(x),'-g',linewidth=2.,label='-f')
		
		plt.plot(x,int_f(x),'-g',linewidth=2.,label='int(f)')
		
		plt.legend(loc='best')
		plt.axis([0,1,-2,2])
		plt.show()
		plt.clf()
		return
	
	
	x = np.linspace(0,1,101)
	F, u_p = int_f(x), derivative(u,x,dx=1e-6)
	
	def least_squares(c):
		out = ( (3./8 - F)/c - u_p )**2.
		return np.sum(out)
	
	guess = (1./4)*(3-x)
	sol = minimize(least_squares,guess)
	
	step = 10
	plt.plot(x[::step],a(x)[::step],'*b',linewidth=2,label='a(x)')
	
	plt.plot(x,sol.x,'-b',linewidth=2,label='estimate of a')
	plt.plot(x,F,'-g',label='integral of f')
	plt.legend(loc='best')
	plt.show()
	plt.clf()
	return 



def exercise2():
	epsilon = .8
	def f(x):
		out = -np.ones(x.shape)
		return out
	
	
	def a(x):
		out = (x+1.)/(1. + epsilon**(-1.)*np.cos(x/epsilon**2.))
		return out
	
	
	def u(x):
		return x + 1. + epsilon*np.sin(x/epsilon**2.)
	
	
	def int_f(x):
		# out = np.zeros(x.shape)
		# antiderivative here is -x
		out = -1.*x
		return out
	
	
	def plot_example_data():
		x = np.linspace(0,1,2000)
		# plt.plot(x,u(x),'-k',linewidth=2.,label='u')
		# plt.plot(x,derivative(u,x,dx=1e-6),'-r',linewidth=2.,label="u'")
		# plt.plot(x,2*x + 1./2,'*r',label="u'")
		
		plt.plot(x,a(x),'-b',linewidth=2.,label='a')
		plt.plot(x,(1. + epsilon**(-1.)*np.cos(epsilon**(-2.)) )**(-1.)*x,'-g',label='guess for a')
		
		# plt.plot(x,-f(x),'-g',linewidth=2.,label='-f')
		# plt.plot(x,int_f(x),'-g',linewidth=2.,label='int(f)')
		
		plt.legend(loc='best')
		# plt.axis([0,1,-2,2])
		plt.show()
		plt.clf()
		return
	
	# plot_example_data()
	# import sys; sys.exit()
	x = np.linspace(0,1,101)
	F, u_p = int_f(x), derivative(u,x,dx=1e-6)
	
	def least_squares(c):
		out = ( (1. - F)/c - u_p )**2.
		return np.sum(out)
	
	guess = (a(1)-a(0))*x + 1
	sol = minimize(least_squares,guess)
	
	
	step = 5
	# plt.plot(x[::step],a(x)[::step],'*b',linewidth=2,label='a(x)')
	# plt.plot(x,sol.x,'-b',linewidth=2,label='estimate of a')
	# plt.legend(loc='best')
	plt.plot(x,sol.x,'-b',linewidth=2,label='estimate of a')
	plt.show()
	plt.clf()
	return 


# # This code finds where the singularity in epsilon occurs for exercise 2.
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import root
# 
# domain = np.linspace(0,1,2000)
# 
# def y(x): 
# 	out = x + np.cos(x**(-2))
# 	return out
# 
# sol = root(y,.65)
# print sol.x
# plt.plot(domain,y(domain),'-k',linewidth=2.0)
# plt.plot(sol.x[0],0.,'*k')
# plt.show()


if __name__ == "__main__":
	# example_text()
	# example()
	exercise2()