from __future__ import division
import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve, cg

def general_secondorder_ode_fd(func,a1,a2,a3,a=0.,b=1.,alpha=1.,beta=3.,N=5):
	# A Simple Finite Difference Scheme to solve BVP's of the form 
	# a1(x)u''(x) + a2(x)u'(x) + a3(x)u(x) = f(x), x \in [a,b]
	# u(a) = alpha
	# u(b) = beta
	# (Dirichlet boundary conditions)
	# 
	# U_0 = alpha, U_1, U_2, ..., U_m, U_{m+1} = beta
	# We use m+1 subintervals, giving m algebraic equations
    m = N-1
    h = (b-a)/(m+1.)         # Here we form the diagonals
    D0,D1,D2,diags = np.zeros((1,m)), np.zeros((1,m)), np.zeros((1,m)), np.array([0,-1,1])
    for j in range(1,D1.shape[1]):
		xj = a + (j+1)*h
		D0[0,j] = h**2.*a3(xj)-2.*a1(xj)
		D1[0,j] = a1(xj)+h*a2(xj)/2.
		D2[0,j-1] = a1(xj)-h*a2(xj)/2.
    xj = a + 1.*h
    D0[0,0] = h**2.*a3(xj)-2.*a1(xj)
	
    # Here we create the matrix A
    data = np.concatenate((D0,D2,D1),axis=0) # This stacks up rows
    A=h**(-2.)*spdiags(data,diags,m,m).asformat('csr')
	
	# Here we create the vector B
    B = np.zeros(m+2)
    for j in range(2,m):
        B[j] = func(a + j*h)
    xj = a+1.*h
    B[0], B[1] = alpha, func(xj)-alpha *( a1(xj)*h**(-2.) - a2(xj)*h**(-1)/2. )
    xj = a+m*h
    B[-1], B[-2]  = beta, func(xj)-beta*( a1(xj)*h**(-2.) + a2(xj)*h**(-1)/2. )
	
    # Here we solve the equation AX = B and return the result
    B[1:-1] = spsolve(A,B[1:-1])
    return np.linspace(a,b,m+2), B



def poisson_square(a1,b1,c1,d1,n,bcs, source): 
	#n = number of subintervals
	# We discretize in the x dimension by a1 = x_0 < x_1< ... < x_n=b1, and 
	# we discretize in the y dimension by c1 = y_0 < y_1< ... < y_n=d1. 
	# This means that we have interior points {x_1, ..., x_{n-1}}\times {y_1, ..., y_{n-1}}
	# or {x_1, ..., x_m}\times {y_1, ..., y_m} where m = n-1. 
	# In Python, this is indexed as {x_0, ..., x_{m-1}}\times {y_0, ..., y_{m-1}}
	# We will have m**2 pairs of interior points, and m**2 corresponding equations.
	# We will organize these equations by their y coordinates: all equations centered 
	# at (x_i, y_0) will be listed first, then (x_i, y_1), and so on till (x_i, y_{m-1})
	delta_x, delta_y, h, m = (b1-a1)/n, (d1-c1)/n, (b1-a1)/n, n-1
	
	####    Here we construct the matrix A    ####
	##############################     Slow            #################################
	#     D, diags = np.ones((1,m**2)), np.array([-m,m])
	#     data = np.concatenate((D, D),axis=0) 
	#     A = h**(-2)*spdiags(data,diags,m**2,m**2).asformat('lil')
	#     D = np.ones((1,m))
	#     diags, data = np.array([0,-1,1]), np.concatenate((-4.*D,D,D),axis=0)
	#     temp = h**(-2)*spdiags(data,diags,m,m).asformat('lil')
	#     for i in xrange(m): A[i*m:(i+1)*m,i*m:(i+1)*m] = temp
	
	##############################     Much Faster      ################################
	D1,D2,D3 = -4*np.ones((1,m**2)), np.ones((1,m**2)), np.ones((1,m**2)) 
	Dm1, Dm2 = np.ones((1,m**2)), np.ones((1,m**2))
	for j in range(0,D2.shape[1]):
		if (j%m)==m-1: 
			D2[0,j]=0
		if (j%m)==0: 
			D3[0,j]=0
	diags = np.array([0,-1,1,-m,m])
	data = np.concatenate((D1,D2,D3,Dm1,Dm2),axis=0) # This stacks up rows
	A = 1./h**2.*spdiags(data, diags, m**2,m**2).asformat('csr') # This appears to work correctly
	
	####    Here we construct the vector b    ####
	b, Array = np.zeros(m**2), np.linspace(0.,1.,m+2)[1:-1]
	# In the next line, source represents the inhomogenous part of Poisson's equation
	for j in xrange(m): b[j*m:(j+1)*m] = source(a1+(b1-a1)*Array, c1+(j+1)*h*np.ones(m) )
	
    # In the next four lines, bcs represents the Dirichlet conditions on the boundary
#     y = c1+h, d1-h
	b[0:m] = b[0:m] - h**(-2.)*bcs(a1+(b1-a1)*Array,c1*np.ones(m))
	b[(m-1)*m : m**2] = b[(m-1)*m : m**2] - h**(-2.)*bcs(a1+(b1-a1)*Array,d1*np.ones(m))
#     x = a1+h, b1-h
	b[0::m] = b[0::m] - h**(-2.)*bcs(a1*np.ones(m),c1+(d1-c1)*Array) 
	b[(m-1)::m] = b[(m-1)::m] - h**(-2.)*bcs(b1*np.ones(m),c1+(d1-c1)*Array)
	
    ####    Here we solve the system A*soln = b    ####
	soln = spsolve(A,b) # Using the conjugate gradient method: (soln, info) = cg(A,b)
	
	z = np.zeros((m+2,m+2) ) 
	for j in xrange(m): z[1:-1,j+1] = soln[j*m:(j+1)*m]
	
	x, y = np.linspace(a1,b1,m+2), np.linspace(c1,d1,m+2)
	z[:,0], z[:,m+1]  = bcs(x,c1*np.ones(len(x)) ), bcs(x,d1*np.ones(len(x)) )
	z[0,:], z[m+1,:] = bcs(a1*np.ones(len(x)),y), bcs(b1*np.ones(len(x)),y)
	return z



