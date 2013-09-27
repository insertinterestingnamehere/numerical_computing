import matplotlib
# matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
import solution


def Fig1(): 
# Plot #1: The solution of y'=y-2x+4, y(0)=0, is 
# y(x) = -2 + 2x + (ya + 2)e^x. This code plots the solution for 0<x<2,
# and then plots the approximation given by Euler's method
# Text Example (f1).
	a, b, ya = 0.0, 2.0, 0.0

	x = np.linspace(a,b,11)
	Y_E = solution.Euler(solution.ode_f1,a,b,10,ya) 
	plt.plot(x, Y_E, 'b-',label="h = 0.2")

	x = np.linspace(a,b,21)
	Y_E = solution.Euler(solution.ode_f1,a,b,20,ya) 
	plt.plot(x, Y_E, 'g-',label="h = 0.1")

	x = np.linspace(a,b,41)
	Y_E = solution.Euler(solution.ode_f1,a,b,40,ya) 
	plt.plot(x, Y_E, 'r-',label="h = 0.05")

	x1 = np.linspace(0,2,200); k =int(200/40)

	plt.plot(x1[::k], solution.function(x1,solution.f1,0.0)[::k], 'k*-',label="Solution") # The solution 
	plt.plot(x1[k-1::k], solution.function(x1,solution.f1,0.0)[k-1::k], 'k-') # The solution 

	plt.legend(loc='best')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.savefig('Fig1.pdf')
	plt.clf()

	return 

def Fig2(): 
# Plot #2: Integral curves for f1(x). Text Example (f1).
	a , b, n = 0.0,  1.6,  200
	h = (b-a)/n
	k =int(n/40)

	x = np.linspace(a,b,n+1); 

	plt.plot(x, solution.function(x,solution.f1,0.0), 'k-')
	plt.plot(x, solution.function(x,solution.f1,-1.0), 'k-')
	plt.plot(x[::k], solution.function(x,solution.f1,-2.0)[::k], 'k*-', label='Particular solution for 'r"$y'-y=-2x+4 $.")
	plt.plot(x, solution.function(x,solution.f1,-3.0), 'k-')
	plt.plot(x, solution.function(x,solution.f1,-4.0), 'k-')
	plt.plot(x, solution.function(x,solution.f1,-5.0), 'k-')

	plt.legend(loc='best')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.savefig('Fig2.pdf')
	plt.clf()

	return


def Fig3(): 
# Plot #3: Integral curves for f2(x).
	a , b, n = 0.0,  1.6,  200
	x = np.linspace(a,b,n+1); 
	k =int(n/20)

	plt.plot(x, solution.function(x,solution.f2,0.0), 'k-')
	plt.plot(x, solution.function(x,solution.f2,2.0), 'k-')
	plt.plot(x[::k], solution.function(x,solution.f2,4.0)[::k], 'k*-', label='Particular solution for 'r"$y' +y =  - 2x + 2 $.")
	plt.plot(x, solution.function(x,solution.f2,6.0), 'k-')
	plt.plot(x, solution.function(x,solution.f2,8.0), 'k-')

	plt.legend(loc='best')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.savefig('Fig3.pdf')
	plt.clf()

	return 

def Fig4():
# Plot #4: Integral curves for y' = sin y using dopri5 
	a, b, n = 0.0, 5.0, 50
	k = n//10
	x = np.linspace(a,b,n+1)
	h = (b-a)/n

	def dopri5_integralcurves(ya): 
		test1 = ode(solution.ode_f3).set_integrator('dopri5',atol=1e-7,rtol=1e-8,nsteps=500) 
		y0 = ya; x0 = a; test1.set_initial_value(y0,x0) 
		Y = np.zeros(x.shape); Y[0] = y0
		for j in range(1,len(x)): 
			test1.integrate(x[j])
			Y[j]= test1.y
		return Y

	plt.plot(x[::k], dopri5_integralcurves(5.0*np.pi/2.0)[::k], 'k-')
	plt.plot(x[::k], dopri5_integralcurves(3.0*np.pi/2.0)[::k], 'k-')
	plt.plot(x[::k], dopri5_integralcurves(7.0*np.pi/4.0)[::k], 'k-')
	plt.plot(x[::k], dopri5_integralcurves(0.0*np.pi/2.0)[::k], 'k-')
	plt.plot(x[::k], dopri5_integralcurves(-np.pi)[::k], 'k*-',label='Equilibrium solutions')
	plt.plot(x[::k], dopri5_integralcurves(np.pi)[::k], 'k*-')
	plt.plot(x[::k], dopri5_integralcurves(2*np.pi)[::k], 'k*-')
	plt.plot(x[::k], dopri5_integralcurves(3*np.pi)[::k], 'k*-')
	plt.plot(x[::k], dopri5_integralcurves(np.pi/4.0)[::k], 'k-')
	plt.plot(x[::k], dopri5_integralcurves(np.pi/2.0)[::k], 'k-')
	plt.plot(x[::k], dopri5_integralcurves(-np.pi/2.0)[::k], 'k-')

	plt.legend(loc='best')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.savefig('Fig4.pdf')
	plt.clf()
	
	return 
	
Fig1()
Fig2()
Fig3()
Fig4()
