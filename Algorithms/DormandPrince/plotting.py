#! /usr/bin/env python
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


def weightloss_calculator(age, sex, H, BW, T, (PAL,EI) ):
	# Initial stats
	# Age (y), Gender	('male' or 'female'), Height (m), Body Weight (kg)	
	# Time (d)
	# 	
	# Diet/Lifestyle Change
	# (PAL, EI) = Future Physical Activity Level and Energy Intake
	#
	# Call the IVP Solver     
	########################################
	from solution import fat_mass, compute_weight_curve
	F = fat_mass(BW,age,H,sex)
	L = BW-F
	t,y = compute_weight_curve(F,L,T,EI,PAL)
	
	# Plot the Results
	####################################
	fig, ax = plt.subplots()
	plt.plot(t,2.2*y[:,0],'-b',label='Fat',linewidth=2.0)
	plt.plot(t,2.2*y[:,1],'-g',label='Lean',linewidth=2.0)
	plt.plot(t,2.2*(y[:,0]+y[:,1]),'-r',label='Total',linewidth=2.0)
	plt.legend(loc=1)# Upper right placement
	plt.xlabel('days',fontsize=16)
	plt.ylabel('lbs',fontsize=16)
	plt.axis([0, np.max(t),20, 180])
	
	plt.plot(t, 2.2*25*H**2*np.ones(t.shape),'-.k')  # High end of normal weight range
	plt.plot(t, 2.2*20*H**2*np.ones(t.shape),'-.k')  # Low end of normal weight range
	
	from matplotlib.ticker import MultipleLocator
	majorLocator   = MultipleLocator(200)
	ax.xaxis.set_major_locator(majorLocator)
	plt.savefig('weightloss.pdf')
	plt.show()
	return 


def Exercise5():
	age, sex =  38. , 'female'
	H, BW    =  1.73, 72.7
	T		 =  5*7*52.   # Long time frame       
	
	# PAL, EI = 1.5, 2025.
	PALf, EIf = 1.5, 2025
	def EI(t): return EIf
	
	def PAL(t): return PALf
	
	weightloss_calculator(age, sex, H, BW, T, (PAL,EI) )
	return 


def Exercise6():
	age, sex =  38. , 'female'
	H, BW    =  1.73, 72.7
	T		 =  5*7*52.   # Long time frame       
	
	PALf, EIf = 1.4, 1850
	def EI(t): return EIf
	
	def PAL(t): return PALf
	
	weightloss_calculator(age, sex, H, BW, T, (PAL,EI) )	
	return 


def Exercise7():
	age, sex =  38. , 'female'
	H, BW    =  1.73, 72.7
	T		 =  16*7*15.   # Long time frame       
	
	PALf, EIf = 1.5, 2025
	def EI(t): 
		if t<16*7*1.: return 1600
		else: return EIf
	
	
	def PAL(t): 
		if t<16*7*1.: return 1.7
		else: return PALf
	
	
	weightloss_calculator(age, sex, H, BW, T, (PAL,EI) )	
	return



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


# Exercise5()
# Exercise6()
Exercise7()













