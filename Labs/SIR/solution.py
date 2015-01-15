#! /usr/bin/env python
from __future__ import division
from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt


def SIR(a,b,beta, gamma,ya): 
	def SIR_ode(t,y): 
		return np.array([ -beta*y[0]*y[1], beta*y[0]*y[1]-gamma*y[1], gamma*y[1] ])
	
	example = ode(SIR_ode).set_integrator('dopri5',atol=1e-8,rtol=1e-8) 
	example.set_initial_value(ya,a) 
	
	dim, t = 3, np.linspace(a,b,501)
	Y = np.zeros((len(t),dim))
	Y[0,:] = ya
	for j in range(1,len(t)): 
		Y[j,:] = example.integrate(t[j])  
	
	return t, Y

# 
# def SIS(a,b,beta, gamma,ya): 
# 	def SIS_ode(t,y): return np.array([ -beta*y[0]*y[1] +  gamma*y[1], 
# 										 beta*y[0]*y[1] - gamma*y[1]    ])
# 	
# 	example = ode(SIS_ode).set_integrator('dopri5',atol=1e-8,rtol=1e-8) 
# 	example.set_initial_value(ya,a) 
# 	
# 	dim, t = 2, np.linspace(a,b,501)
# 	Y = np.zeros((len(t),dim))
# 	Y[0,:] = ya
# 	for j in range(1,len(t)): Y[j,:] = example.integrate(t[j])  
# 	
# 	plt.plot(t,Y[:,0],'-k',label='Susceptible')
# 	plt.plot(t,Y[:,1],'-r',label='Infectious')
# 	plt.axis([a,b,-.1,1.1])
# 	plt.legend(loc=1)
# 	plt.xlabel('T (days)',fontsize=16)
# 	plt.ylabel('Proportion of Population',fontsize=16)
# 	plt.show()
# 	# plt.clf()
# 	return t, Y
# 
# 
# def SIRS(a,b,beta, gamma, mu, f,ya): 
# 	def SIRS_ode(t,y): return np.array([ -beta*y[0]*y[1]+ mu*(1-y[0]) + f*y[2], 
# 										  beta*y[0]*y[1] - (gamma + mu) * y[1], 
# 										  gamma*y[1] -mu*y[2] - f*y[2]						])
# 	
# 	example = ode(SIRS_ode).set_integrator('dopri5',atol=1e-8,rtol=1e-8) 
# 	example.set_initial_value(ya,a) 
# 	
# 	dim, t = 3, np.linspace(a,b,501)
# 	Y = np.zeros((len(t),dim))
# 	Y[0,:] = ya
# 	for j in range(1,len(t)): Y[j,:] = example.integrate(t[j])  
# 	
# 	plt.plot(t,Y[:,0],'-k',label='Susceptible')
# 	plt.plot(t,Y[:,2],'-b',label='Recovered')
# 	plt.plot(t,Y[:,1],'-r',label='Infected')
# 	plt.axis([a,b,-.1,1.1])
# 	plt.legend(loc=1)
# 	plt.xlabel('T (days)',fontsize=16)
# 	plt.ylabel('Proportion of Population',fontsize=16)
# 	plt.show()
# 	# plt.clf()
# 	return t, Y
# 
# 






###########################################################################
######      THE SIR MODEL
######      beta  = average number of infectious contacts per day
######      gamma = 1./(average length of time in the infectious phase)


# beta, gamma = 2., 1.
# a, b, ya = 0., 50., np.array([1.-(1.67e-6), 1.67e-6,0.])

# beta, gamma = .340, .333
# a, b, ya = 0., 1600., np.array([1.-(6.25e-7), 6.25e-7,0.])

# t,Y = SIR(a,b,beta, gamma,ya)
# print "The Maximum fraction of the population that will be infected simultaneously is", max(Y[:,1])                    
###########################################################################


###########################################################################
######      THE SIS MODEL
######		Here we suppose N is the total population without infection. We do 
######		suppose that S+I=1
######      beta  = average number of infectious contacts per day
######      gamma = 1./(average length of time in the infectious phase)

# beta, gamma = 3./10., 1./4.  
# a, b, ya = 0., 400., np.array([1.-(1.667e-6), 1.667e-6])
# t,Y = SIS(a,b,beta, gamma,ya)
# print "The Maximum fraction of the population that will be infected simultaneously is", max(Y[:,1])
###########################################################################


###########################################################################
######      THE SIRS MODEL
######		Here we suppose N is the total population without infection. We do 
######		suppose that S+I+R=1
######      beta  = average number of infectious contacts per day
######      gamma = 1./(average length of time in the infectious phase)

# beta, gamma = 3./10., .7/4.  
# mu, f= .1, .3
# a, b, ya = 0., 3600., np.array([1.-50*(1.667e-6), 50*1.667e-6,0.])
# t,Y = SIRS(a,b,beta, gamma, mu, f, ya)
# print "The Maximum fraction of the population that will be infected simultaneously is", max(Y[:,1])
###########################################################################





