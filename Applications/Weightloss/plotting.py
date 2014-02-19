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
	T		 =  16*7*2.   # Long time frame       
	
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

# t,Y = Exercise2_4(a,b,beta, gamma,ya)
# print "The Maximum fraction of the population that will be infected simultaneously is", max(Y[:,1])                    


# Exercise5()
# Exercise6()
# Exercise7()


# Lotka-Volterra Predator Prey Model
# U_t = U(1-V)
# V_t = alpha V(U-1)
# 
# Logistic Predator Prey Model
# U_t = U(1-U-V)
# V_t = alpha V(U-beta)



def Example_Lotka_Volterra():
	a,b  	= 0., 30. 					# (Nondimensional) Time interval 
										# for one 'period' 
	alpha 	= 1./3              		# Nondimensional parameters
	dim, ya = 2, np.array([3/4., 3/4.]) # dimension of the system / 
										# initial conditions
	
	def Lotka_Volterra(x,y):
		return np.array([y[0]*(1. - y[1]), alpha*y[1]*(y[0] - 1.)])
	
	from solution import RK4
	subintervals=500
	Y = RK4( Lotka_Volterra,a,b,subintervals,ya,dim)
	
	# Plot the direction field
	Y1,Y2 = np.meshgrid( np.arange(0,4.5,.2), np.arange(0,4.5,.2),
										sparse=True)
	U,V = Lotka_Volterra(0,(Y1,Y2))
	Q = plt.quiver( Y1[::3, ::3], Y2[::3, ::3], 
					 U[::3, ::3],  V[::3, ::3],
	            	pivot='mid', color='b', units='dots',width=3. )
	# Plot the 2 Equilibrium points
	plt.plot(1,1,'ok',markersize=8); plt.plot(0,0,'ok',markersize=8) 
	# Plot the solutions in phase space
	# Y = RK4( Lotka_Volterra,a,b,subintervals,np.array([3/4., 3/4.]),dim)
	plt.plot(Y[:,0], Y[:,1],'-k',linewidth=2.0)	
	# Y = RK4( Lotka_Volterra,a,b,subintervals,np.array([1/2., 3/4.]),dim)
	# plt.plot(Y[:,0], Y[:,1],'-k',linewidth=2.0)
	# # Y = RK4( Lotka_Volterra,a,b,subintervals,np.array([1/4., 1/3.]),dim)
	# # plt.plot(Y[:,0], Y[:,1],'-k',linewidth=2.0)
	# Y = RK4( Lotka_Volterra,a,b,subintervals,np.array([1/16., 3/4.]),dim)
	# plt.plot(Y[:,0], Y[:,1],'-k',linewidth=2.0)
	# Y = RK4( Lotka_Volterra,a,b,subintervals,np.array([1/40., 3/4.]),dim)
	# plt.plot(Y[:,0], Y[:,1],'-k',linewidth=2.0)
	# plt.plot(Y[::10,0], Y[::10,1],'*b')
	
	plt.axis([-.5, 4.5, -.5, 4.5])
	plt.title("Phase Portrait of the " + 
					"Lotka-Volterra Predator-Prey Model")
	plt.xlabel('Prey',fontsize=15); plt.ylabel('Predators',fontsize=15)
	plt.savefig("Lotka_Volterra_Phase_Portrait.pdf")
	plt.clf()
	Y = RK4( Lotka_Volterra,a,2*b,2*subintervals,ya,dim)
	plt.plot(np.linspace(a,2*b,2*subintervals+1), Y[:,0],'-b',linewidth=2.0)
	plt.plot(np.linspace(a,2*b,2*subintervals+1), Y[:,1],'-g',linewidth=2.0)
	plt.savefig("Lotka_Volterra.pdf")
	plt.clf()
	# plt.show()
	return



def Exercise_Lotka_Volterra():
	a,b  	= 0., 30. 					# (Nondimensional) Time interval 
										# for one 'period' 
	alpha 	= 1./3              		# Nondimensional parameters
	dim, ya = 2, np.array([3/4., 3/4.]) # dimension of the system / 
										# initial conditions
	
	def Lotka_Volterra(x,y):
		return np.array([y[0]*(1. - y[1]), alpha*y[1]*(y[0] - 1.)])
	
	from solution import RK4
	subintervals=500
	Y = RK4( Lotka_Volterra,a,b,subintervals,ya,dim)
	
	# Plot the direction field
	Y1,Y2 = np.meshgrid( np.arange(0,4.5,.2), np.arange(0,4.5,.2),
										sparse=True)
	U,V = Lotka_Volterra(0,(Y1,Y2))
	Q = plt.quiver( Y1[::3, ::3], Y2[::3, ::3], 
					 U[::3, ::3],  V[::3, ::3],
	            	pivot='mid', color='b', units='dots',width=3. )
	# Plot the 2 Equilibrium points
	plt.plot(1,1,'ok',markersize=8); plt.plot(0,0,'ok',markersize=8) 
	# Plot the solutions in phase space
	plt.plot(Y[:,0], Y[:,1],'-k',linewidth=2.0)	
	Y = RK4( Lotka_Volterra,a,b,subintervals,np.array([1/2., 3/4.]),dim)
	plt.plot(Y[:,0], Y[:,1],'-k',linewidth=2.0)
	Y = RK4( Lotka_Volterra,a,b,subintervals,np.array([1/16., 3/4.]),dim)
	plt.plot(Y[:,0], Y[:,1],'-k',linewidth=2.0)
	Y = RK4( Lotka_Volterra,a,b,subintervals,np.array([1/40., 3/4.]),dim)
	plt.plot(Y[:,0], Y[:,1],'-k',linewidth=2.0)
	plt.plot(Y[::10,0], Y[::10,1],'*b')
	
	plt.axis([-.5, 4.5, -.5, 4.5])
	plt.title("Phase Portrait of the " + 
					"Lotka-Volterra Predator-Prey Model")
	plt.xlabel('Prey',fontsize=15); plt.ylabel('Predators',fontsize=15)
	# plt.savefig("Lotka_Volterra_Phase_Portrait.pdf")
	plt.clf()
	# plt.show()
	return


def Exercise_Logistic():
	# y[0], y[1] = Prey, Predator populations
	a,b = 0., 40.
	dim=2
	ya1 = np.array([1/3., 1/3.])
	ya2 = np.array([2.5/5., 1/5.])
	alpha, beta = 1.,.3
	
	def Logistic(x,y):
		return np.array([y[0]*(1-y[0]-y[1]), alpha*y[1]*(y[0] - beta)])
	
	example1 = ode(Logistic).set_integrator('dopri5',atol=1e-10) 
	example1.set_initial_value(ya1,a) 
	example2 = ode(Logistic).set_integrator('dopri5',atol=1e-10) 
	example2.set_initial_value(ya2,a)
	
	t = np.linspace(a,b,201)
	Y1 = np.zeros((len(t),dim)); Y1[0,:] = ya1
	Y2 = np.zeros((len(t),dim)); Y2[0,:] = ya2
	
	for j in range(1,len(t)): 
		Y1[j,:] = example1.integrate(t[j])
		Y2[j,:] = example2.integrate(t[j])
	
	plt.plot(Y1[:,0], Y1[:,1],'-k',linewidth=1.5); # plt.plot(Y1[::5,0], Y1[::5,1],'*g')
	plt.plot(Y2[:,0], Y2[:,1],'-k',linewidth=1.5); # plt.plot(Y2[::5,0], Y2[::5,1],'*g')
	
	R,S = np.meshgrid( np.arange(0.,1.35,.1),np.arange(0.,1.35,.1) ,sparse=True)
	U,V = Logistic(0,(R,S))
	Q = plt.quiver( R[::2, ::2], S[::2, ::2], U[::2, ::2], V[::2, ::2],
	            pivot='mid', color='green', units='dots' ,width=3.)
	
	plt.plot(beta,1-beta,'ok',markersize=6)
	plt.plot(1,0,'ok',markersize=6)
	plt.plot(0,0,'ok',markersize=6)
	# plt.plot( R[::2, ::2], S[::2, ::2], 'k.')
	plt.axis([-.1, 1.3, -.1, 1.3])
	plt.title("Phase Portrait of Logistic Predator-Prey Model")
	plt.xlabel('Prey',fontsize=15); plt.ylabel('Predators',fontsize=15)
	plt.show()
	return




Exercise_Logistic()
Example_Lotka_Volterra()

Exercise_Lotka_Volterra()









