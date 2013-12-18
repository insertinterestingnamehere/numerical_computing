from scipy.optimize import newton
import numpy as np

def Euler(func,a,b,n,y0):
	x = np.linspace(a,b,n+1); Y = np.zeros(x.shape); Y[0] = y0
	h = (b-a)/n
	for j in range(0,len(x)-1): 
		Y[j+1] = Y[j] + h*func( x[j],Y[j] )
	return Y

	
def backward_Euler(func,a,b,n,y0):
	x = np.linspace(a,b,n+1); Y = np.zeros(x.shape); Y[0] = y0
	h = (b-a)/n
	for j in range(0,len(x)-1): 
		g = lambda y: y-Y[j]-h*func(x[j+1],y)
		Y[j+1] = newton(g, Y[j], fprime=None, args=(), tol=1.0e-08, maxiter=50)
	return Y


def modified_Euler(func,a,b,n,y0):
	x = np.linspace(a,b,n+1); Y = np.zeros(x.shape); Y[0] = y0
	h = (b-a)/n
	for j in range(0,len(x)-1): 
		Y[j+1] = Y[j] + (h/2.0)*(func( x[j],Y[j] ) + 
			func( x[j+1],Y[j] +h*func( x[j],Y[j] )) )
	return Y


def Midpoint(func,a,b,n,y0):
	x = np.linspace(a,b,n+1); Y = np.zeros(x.shape); Y[0] = y0
	h = (b-a)/n
	for j in range(0,len(x)-1): 	
		Y[j+1] = Y[j]+h*func(x[j] + h/2,Y[j] + (h/2)*func(x[j],Y[j]) )
	return Y


def Heun(func,a,b,n,y0):
	x = np.linspace(a,b,n+1); Y = np.zeros(x.shape); Y[0] = y0
	h = (b-a)/n
	for j in range(0,len(x)-1): 
		Y[j+1] = Y[j] + (h/4.0)*(func(x[j],Y[j])  + 
			3*func(x[j]+(2/3.0)*h,Y[j] + (2/3.0)*h*func(x[j],Y[j]) )  )
	return Y


# Implementation of the Runge Kutta fourth order method
def Runge_Kutta_4(func,a,b,n,y0,dim):
	x = np.linspace(a,b,n+1); 
	Y = np.zeros((len(x),dim)); 
	Y[0,:] = y0
	
	h = 1.*(b-a)/n
	for j in range(0,len(x)-1): 
		k1 = h*func(x[j],Y[j,:])
		k2 = h*func(x[j]+h/2.,Y[j,:]+(1/2.)*k1)
		k3 = h*func(x[j]+h/2.,Y[j,:]+(1/2.)*k2)
		k4 = h*func(x[j+1],Y[j,:]+k3)
		Y[j+1,:] = Y[j,:] + (1/6.)*(k1 + 2*k2 + 2*k3 + k4)
	return Y


# Definition of f(y) in the 2 dimensional ode y' = f(y) for a harmonic oscillator
def harmonic_oscillator_ode(x,y,m,gamma,k,F): 
	return np.array([y[1] ,-1.*gamma/m*y[1]-1.*k/m*y[0] + F(x) ])
	





def ode_f(x,y): return np.array([1.*y + 3. - 1.*x])


def ode_f2(x,y): return np.array([-1.*y+6.+2.*x])


def function(x,g,ya):
	Y = np.zeros(x.shape)
	for j in range(0,len(x)): Y[j] = g(x[j],ya)
	return Y



