#! /usr/bin/env python
from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt



def Example1():
	a, ya, b = 0., 2., 1.6
	
	def ode_f(t,y):
		out = -1.*y+6.+2.*t
		return np.array([out])
	
	example = ode(ode_f)
	
	example.set_integrator('dopri5',atol=1e-5) 
	example.set_initial_value(ya,a) 
	example.integrate(b)
	return example.integrate(b)[0]


def Example2(): 
	a, ya, b = 0., 2., 1.6
	
	def ode_f(t,y):
		out = -1.*y+6.+2.*t
		return np.array([out])
	
	example = ode(ode_f).set_integrator('dopri5',atol=1e-5) 
	example.set_initial_value(ya,a) 
	
	t = np.linspace(a,b,51)
	dim=1
	Y = np.zeros((len(t),dim)); Y[0,:] = ya
	
	for j in range(1,len(t)): 
		Y[j,:] = example.integrate(t[j])  
	
	plt.plot(t,Y[:,0],'-k')
# 	plt.show()
	plt.clf()
	
	return 


def Example3(): 
	a, b, ya = 0., 10., 0.
	m = 3.
	def ode_f(t,y):
		out = m-(1/m)*(y+.5*t**2.)
		return np.array([out])
# 	Exact Solution is y = m*t-.5*t**2.
	
	example = ode(ode_f).set_integrator('dopri5',atol=1e-5) 
	example.set_initial_value(ya,a) 
	
	t = np.linspace(a,b,51)
	dim=1
	Y = np.zeros((len(t),dim)); Y[0,:] = ya
	
	for j in range(1,len(t)): 
		Y[j,:] = example.integrate(t[j])  
	
	plt.plot(t,Y[:,0],'-k')
# 	plt.show()
	plt.clf()
	
	return 


def Exercise1(): 
	a, b, ya = 0., 16.,np.array([0,1,-2])
	
	def ode_f(t,y):
		out = np.array([y[1],y[2], -.2*(y[1] + 2.*y[0])])
		return out
	
	example = ode(ode_f).set_integrator('dopri5',atol=1e-8) 
	example.set_initial_value(ya,a) 
	
	t = np.linspace(a,b,201)
	dim=3
	Y = np.zeros((len(t),dim)); Y[0,:] = ya
	
	for j in range(1,len(t)): 
		Y[j,:] = example.integrate(t[j])  
	
	plt.plot(t,Y[:,0],'-k')
	plt.show()
	plt.clf()
	return 


def Exercise2(): 
	beta = .5#.340		# average number of infectious contacts per day
	gamma = .25#.333	# 1./(average length of time in the infectious phase)	
				#b = 1600
	def SIR_ode(t,y): 
		return np.array([ -beta*y[0]*y[1], beta*y[0]*y[1]-gamma*y[1], gamma*y[1] ])
	
	a, b = 0., 100.
	ya = np.array([1.-(6.25e-7), 6.25e-7,0.])
	example = ode(SIR_ode).set_integrator('dopri5',atol=1e-8,rtol=1e-8) 
	example.set_initial_value(ya,a) 
	
	t = np.linspace(a,b,501)
	dim=3
	Y = np.zeros((len(t),dim)); Y[0,:] = ya
	
	for j in range(1,len(t)): 
		Y[j,:] = example.integrate(t[j])  
	
	plt.plot(t,Y[:,0],'-k',label='Susceptible')
	plt.plot(t,Y[:,2],'-b',label='Recovered')
	plt.plot(t,Y[:,1],'-r',label='Infected')
	plt.axis([a,b,-.1,1.1])
	plt.legend(loc=1)
	plt.xlabel('T (days)',fontsize=16)
	plt.ylabel('Proportion of Population',fontsize=16)
	plt.show()
# 	plt.clf()
	return


def Exercise3(): 
# 	beta = .340		# beta = average number of infectious contacts per day
# 	gamma = .333	# gamma = 1./(average length of time in the infectious phase)	
				#b = 1600
	beta = 2.#1/1.
	gamma = 1.#1/7.
	a, b = 0., 50.
	
	def SIR_ode(t,y): 
		return np.array([ -beta*y[0]*y[1], beta*y[0]*y[1]-gamma*y[1], gamma*y[1] ])
	
	ya = np.array([1.-(1.67e-6), 1.67e-6,0.])
	example = ode(SIR_ode).set_integrator('dopri5',atol=1e-8,rtol=1e-8) 
	example.set_initial_value(ya,a) 
	
	t = np.linspace(a,b,501)
	dim=3
	Y = np.zeros((len(t),dim)); Y[0,:] = ya
	
	for j in range(1,len(t)): 
		Y[j,:] = example.integrate(t[j])  
	
	plt.plot(t,Y[:,0],'-k',label='Susceptible')
	plt.plot(t,Y[:,2],'-b',label='Recovered')
	plt.plot(t,Y[:,1],'-r',label='Infected')
	plt.axis([a,b,-.1,1.1])
	plt.legend(loc=1)
	plt.xlabel('T (days)',fontsize=16)
	plt.ylabel('Proportion of Population',fontsize=16)
	plt.show()
# 	plt.clf()
	return


def Exercise4(): 
	beta = .3		# beta = average number of infectious contacts per day
	gamma = 1/4.	# gamma = 1./(average length of time in the infectious phase)	
				#b = 1600
				
	a, b = 0., 400.
	
	def SIR_ode(t,y): 
		return np.array([ -beta*y[0]*y[1], beta*y[0]*y[1]-gamma*y[1], gamma*y[1] ])
	
	ya = np.array([1.-(1.67e-6), 1.67e-6,0.])
	example = ode(SIR_ode).set_integrator('dopri5',atol=1e-8,rtol=1e-8) 
	example.set_initial_value(ya,a) 
	
	t = np.linspace(a,b,501)
	dim=3
	Y = np.zeros((len(t),dim)); Y[0,:] = ya
	
	for j in range(1,len(t)): 
		Y[j,:] = example.integrate(t[j])  
	
	plt.plot(t,Y[:,0],'-k',label='Susceptible')
	plt.plot(t,Y[:,2],'-b',label='Recovered')
	plt.plot(t,Y[:,1],'-r',label='Infected')
	plt.axis([a,b,-.1,1.1])
	plt.legend(loc=1)
	plt.xlabel('T (days)',fontsize=16)
	plt.ylabel('Proportion of Population',fontsize=16)
	plt.show()
# 	plt.clf()
	return

# Exercise1()
# Exercise2()
Exercise3()
# Exercise4()
		
# Example1()
# Example2()
# Example3()
