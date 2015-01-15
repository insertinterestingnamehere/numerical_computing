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
    D0,Dp,Dm,diags = np.zeros((1,m)), np.zeros((1,m)), np.zeros((1,m)), np.array([0,-1,1])
    for j in range(1,D0.shape[1]):
		xj = a + (j)*h
		D0[0,j]   = h**2.*a3(xj)-2.*a1(xj)
		Dp[0,j]   = a1(xj)-h*a2(xj)/2.
		Dm[0,j-1] = a1(xj)+h*a2(xj)/2.
    # xj = a + 1.*h
    # D0[0,0] = h**2.*a3(xj)-2.*a1(xj)
	
    # Here we create the matrix A
    data = np.concatenate((D0,Dm,Dp),axis=0) # This stacks up rows
    A=h**(-2.)*spdiags(data,diags,m,m).asformat('csr')
	
	# Here we create the vector B
    B = np.zeros(m+2)
    for j in range(2,m):
        B[j] = func(a + j*h)
    xj = a+1.*h
    B[0], B[1] = alpha, func(xj)-alpha *( a1(xj)*h**(-2.) + a2(xj)*h**(-1)/2. )
    xj = a+m*h
    B[-1], B[-2]  = beta, func(xj)-beta*( a1(xj)*h**(-2.) - a2(xj)*h**(-1)/2. )
	
    # Here we solve the equation AX = B and return the result
    B[1:-1] = spsolve(A,B[1:-1])
    return np.linspace(a,b,m+2), B



# def general_secondorder_ode_fd(func,a1,a2,a3,a=0.,b=1.,alpha=1.,beta=3.,N=5):
# 	# A Simple Finite Difference Scheme to solve BVP's of the form 
# 	# a1(x)u''(x) + a2(x)u'(x) + a3(x)u(x) = f(x), x \in [a,b]
# 	# u(a) = alpha
# 	# u(b) = beta
# 	# (Dirichlet boundary conditions)
# 	# 
# 	# U_0 = alpha, U_1, U_2, ..., U_m, U_{m+1} = beta
# 	# We use m+1 subintervals, giving m algebraic equations
#     m = N-1
#     h = (b-a)/(m+1.)         # Here we form the diagonals
#     D0,D1,D2,diags = np.zeros((1,m)), np.zeros((1,m)), np.zeros((1,m)), np.array([0,-1,1])
#     for j in range(1,D1.shape[1]):
# 		xj = a + (j+1)*h
# 		D0[0,j] = h**2.*a3(xj)-2.*a1(xj)
# 		D1[0,j] = a1(xj)+h*a2(xj)/2.
# 		D2[0,j-1] = a1(xj)-h*a2(xj)/2.
#     xj = a + 1.*h
#     D0[0,0] = h**2.*a3(xj)-2.*a1(xj)
# 	
#     # Here we create the matrix A
#     data = np.concatenate((D0,D2,D1),axis=0) # This stacks up rows
#     A=h**(-2.)*spdiags(data,diags,m,m).asformat('csr')
# 	
# 	# Here we create the vector B
#     B = np.zeros(m+2)
#     for j in range(2,m):
#         B[j] = func(a + j*h)
#     xj = a+1.*h
#     B[0], B[1] = alpha, func(xj)-alpha *( a1(xj)*h**(-2.) - a2(xj)*h**(-1)/2. )
#     xj = a+m*h
#     B[-1], B[-2]  = beta, func(xj)-beta*( a1(xj)*h**(-2.) + a2(xj)*h**(-1)/2. )
# 	
#     # Here we solve the equation AX = B and return the result
#     B[1:-1] = spsolve(A,B[1:-1])
#     return np.linspace(a,b,m+2), B
# 

