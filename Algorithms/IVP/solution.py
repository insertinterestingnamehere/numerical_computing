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

def Runge_Kutta_4(func,a,b,n,y0):
	x = np.linspace(a,b,n+1); Y = np.zeros(x.shape); Y[0] = y0
	h = (b-a)/n
	for j in range(0,len(x)-1): 
		k1 = h*func(x[j],Y[j])
		k2 = h*func(x[j]+h/2.0,Y[j]+(1/2.0)*k1)
		k3 = h*func(x[j]+h/2.0,Y[j]+(1/2.0)*k2)
		k4 = h*func(x[j+1],Y[j]+k3)
		Y[j+1] = Y[j] + (1/6.0)*(k1 + 2*k2 + 2*k3 + k4)
	return Y

def ode_f(x,y): 
	out = 1.0*y + 3.0 - 1.0*x
	return np.array([out])
	
def ode_f1(x,y):
	out = 1.0*y -2.0*x + 4.0
	return np.array([out])
	
def ode_f2(x,y):
	out = -1.0*y+2.0-2.0*x
	return np.array([out])
	
def ode_f3(x,y): 
	out = np.sin(y)
	return np.array([out])

def f1(x,ya):
	out = -2.0 + 2.0*x + (ya + 2.0)*np.exp(x)
	return out
	
def f2(x,ya):
	out = 4.0 + 2.0*x + (ya - 4.0)*np.exp(-x)
	return out

def function(x,g,ya):
	Y = np.zeros(x.shape); 
	for j in range(0,len(x)):
		Y[j] = g(x[j],ya)
	return Y
