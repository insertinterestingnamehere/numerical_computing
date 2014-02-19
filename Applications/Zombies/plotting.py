#! /usr/bin/env python
from __future__ import division
from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt


def Example():
	a, ya, b = 0., 2., 1.6
	def ode_f(t,y): return np.array([-1.*y+6.+2.*t])
	
	
	example = ode(ode_f)
	example.set_integrator('dopri5',atol=1e-5).set_initial_value(ya,a)
	print example.integrate(b)[0]
	
	example = ode(ode_f).set_integrator('dopri5',atol=1e-5) 
	example.set_initial_value(ya,a) 
	
	dim, t = 1, np.linspace(a,b,51)
	Y = np.zeros((len(t),dim))
	Y[0,:] = ya
	for j in range(1,len(t)): Y[j,:] = example.integrate(t[j])  
	
	# plt.plot(t,Y[:,0],'-k',linewidth=2.0)
	# plt.show()
	# plt.clf()
	return t, Y.T[0]


def AnotherExample(): 
	a, b, ya = 0., 10., 0.
	m = 3.
	def ode_f(t,y): return np.array([m-(1/m)*(y+.5*t**2.)])
	
# 	Exact Solution is y = m*t-.5*t**2.
	
	example = ode(ode_f).set_integrator('dopri5',atol=1e-5) 
	example.set_initial_value(ya,a) 
	
	dim, t = 1, np.linspace(a,b,51)
	Y = np.zeros((len(t),dim))
	Y[0,:] = ya
	for j in range(1,len(t)): Y[j,:] = example.integrate(t[j])  
	
	plt.plot(t,Y[:,0],'-k')
	plt.show()
	plt.clf()
	return 


def Exercise1(): 
	a, b, ya = 0., 16.,np.array([0,1,-2])
	
	def ode_f(t,y): return np.array([y[1],y[2], -.2*(y[1] + 2.*y[0])])
	
	example = ode(ode_f).set_integrator('dopri5',atol=1e-8) 
	example.set_initial_value(ya,a) 
	
	dim, t = 3, np.linspace(a,b,201)
	Y = np.zeros((len(t),dim))
	Y[0,:] = ya
	for j in range(1,len(t)): Y[j,:] = example.integrate(t[j])  
	
	plt.plot(t,Y[:,0],'-k')
	plt.show()
	plt.clf()
	return 


def Exercise2_4(a,b,beta, gamma,ya): 
	def SIR_ode(t,y): return np.array([ -beta*y[0]*y[1], beta*y[0]*y[1]-gamma*y[1], gamma*y[1] ])
	
	example = ode(SIR_ode).set_integrator('dopri5',atol=1e-8,rtol=1e-8) 
	example.set_initial_value(ya,a) 
	
	dim, t = 3, np.linspace(a,b,501)
	Y = np.zeros((len(t),dim))
	Y[0,:] = ya
	for j in range(1,len(t)): Y[j,:] = example.integrate(t[j])  
	
	plt.plot(t,Y[:,0],'-k',label='Susceptible')
	plt.plot(t,Y[:,2],'-b',label='Recovered')
	plt.plot(t,Y[:,1],'-r',label='Infected')
	plt.axis([a,b,-.1,1.1])
	plt.legend(loc=1)
	plt.xlabel('T (days)',fontsize=16)
	plt.ylabel('Proportion of Population',fontsize=16)
	plt.show()
# 	plt.clf()
	return t, Y



# Ans =Example()
# print Ans[0]; print Ans[1]
# AnotherExample()

###########################################################################
######      THE SIR MODEL
######      beta  = average number of infectious contacts per day
######      gamma = 1./(average length of time in the infectious phase)
###########################################################################

# beta, gamma = 2., 1.
# a, b, ya = 0., 50., np.array([1.-(1.67e-6), 1.67e-6,0.])

# beta, gamma = .340, .333
# a, b, ya = 0., 1600., np.array([1.-(6.25e-7), 6.25e-7,0.])

# beta, gamma = 0.5, 0.25   # Exercise 2 
# a, b, ya = 0., 100., np.array([1.-(6.25e-7), 6.25e-7,0.])

# beta, gamma = 1., 1./3.   # Exercise 3a
# a, b, ya = 0., 50., np.array([1.-(1.667e-6), 1.667e-6,0.])

# beta, gamma = 1., 1./7.   # Exercise 3b
# a, b, ya = 0., 50., np.array([1.-(1.667e-6), 1.667e-6,0.])

# beta, gamma = 3./10., 1./4.   # Exercise 4
# a, b, ya = 0., 400., np.array([1.-(1.667e-6), 1.667e-6,0.])

###########################################################################

# t,Y = Exercise2_4(a,b,beta, gamma,ya)
# print "The Maximum fraction of the population that will be infected simultaneously is", max(Y[:,1])                    

