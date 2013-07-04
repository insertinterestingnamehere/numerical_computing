import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.misc import derivative

#Problem 1
def newtonsMethod(f, x0, tol=1e-7, df=None):
    if df is None:
        df = lambda x: derivative(f, x, dx=1e-5)
    
    x = x0
    while(sp.absolute(float(f(x))/df(x)) >= tol):
        x -= float(f(x))/df(x)
    return x

#Problem 2
def Problem2():
    def f(x):
        return x**(1./3)
    
    return f(np.random.rand(100))

#Problem 3
def Problem3(npts=800):
    f = lambda x: x**3 - 2*x + 1./2
    df = lambda x: 3*x**2 - 2.
    
    x = np.random.uniform(-2, 2, size=(npts,))
    
    r = [newtonsMethod(f, x0, df=df) for x0 in x]
    y = f(x)
    
    plt.plot(x, r, '.', x, y, '.')
    plt.show()
    
#Problem 4
def newtonsMatrix(f, x, J=None, tol=1e-7):

#    try:
#        ndim = len(f(*inputs)), len(inputs)
#    except TypeError:
#        ndim = 1, len(inputs)
#    
#    jacobian = sp.zeros(ndim)
#    
#    for j in xrange(ndim[1]):
#        jacobian[:,j] = cdiff(func, inputs, vary=[j], accur=4
#        use scipy deriviative with 1e-5
    

	def jacobian(f, x):
	
	    
		def replace( A,a,i):
			R=A.copy() #This line caused me a lot of problems
			R[i]=a
			return R
			
		J = np.zeros((len(x),len(x)))
		for i in range(len(x)):
			for j in range(len(x)):
				#Is there a better way to do a partial derivative?
				J[i,j] = derivative(lambda a: F(replace(x,a,i))[j],x[i])
		return J
		
	if J is None:
	    J = lambda x: jacobian(f, x)
		
	inc = la.solve(J(x), f(x))
	while(np.absolute(inc).max() >= tol):
		x -= inc
		inc = la.solve(J(x), f(x))
	return x		
		
		

