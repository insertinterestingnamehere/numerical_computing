from __future__ import division

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import spdiags, coo_matrix








def ode_fe(func,c=-1.,d=0.,a=0.,b=1.,alpha=1.,beta=3.,x=np.linspace(0.,1.,5+1)):
	# A Simple Finite Element Scheme to solve BVP's of the form 
	# u''(x) + c*u'(x) + d*u(x) = f(x), x \in [a,b]
	# u(a) = alpha
	# u(b) = beta
	# Dirichlet boundary conditions
	# 
	# U_0 = alpha, U_1, U_2, ..., U_m, U_{m+1} = beta
	# We use m+1 subintervals, giving m algebraic equations
	
	N = len(x)
	# print "Number of finite elements is = ", N-1
	# print "Number of basis functions is = ", N
	
	rows, columns, data = np.zeros(3*(N-2)+2), np.zeros(3*(N-2)+2), np.zeros(3*(N-2)+2)
	for j in range(0,N-2):
		rows[3*j:3*(j+1)] = np.array([j+1,j+1,j+1])
		columns[j*3:(j+1)*3] = np.array(range(0,3))+j
	rows[-2:]= np.array([0,N-1])
	columns[-2:]= np.array([0,N-1])
	
	data[-1], data[-2] = 1,1
	for i in range(1,N-1):
		data[3*(i-1)+2] = 1./(x[i+1]-x[i]) + c*1./2. + d*(x[i+1]-x[i])/6. # i, i+1 location
		data[3*(i-1)+1] = -( 1./(x[i+1]-x[i]) + 1./(x[i]-x[i-1]) 
							) + c*0. + d*(x[i+1]-x[i-1])/3. # i, i location
		data[3*(i-1)] = 1./(x[i]-x[i-1]) - c*1./2. + d*(x[i]-x[i-1])/6. # i, i-1 location
	A = coo_matrix((data, (rows,columns)), shape=(N,N))
	
	B = np.zeros(N)
	B[0], B[-1] = alpha, beta
	for j in range(1,N-1):
		B[j]= ( (x[j]-x[j-1])*func((x[j]+x[j-1])/2.)*(1./2.) + 
		    	(x[j+1]-x[j])*func((x[j+1]+x[j])/2.)*(1./2.)   )
		
	solution = spsolve(A.asformat('csr'),B)
	return x, solution




def tridiag_fe(func,c=-1.,d=0.,a=0.,b=1.,alpha=1.,beta=3.,x=np.linspace(0.,1.,5+1)):
	# A Simple Finite Element Scheme to solve BVP's of the form 
	# u''(x) + c*u'(x) + d*u(x) = f(x), x \in [a,b]
	# u(a) = alpha
	# u(b) = beta
	# Dirichlet boundary conditions
	# 
	# U_0 = alpha, U_1, U_2, ..., U_m, U_{m+1} = beta
	# We use m+1 subintervals, giving m algebraic equations
	
	N = len(x)
	# print "Number of finite elements is = ", N-1
	# print "Number of basis functions is = ", N
	
	rows, columns, data = np.zeros(3*(N-2)+2), np.zeros(3*(N-2)+2), np.zeros(3*(N-2)+2)
	for j in range(0,N-2):
		rows[3*j:3*(j+1)] = np.array([j+1,j+1,j+1])
		columns[j*3:(j+1)*3] = np.array(range(0,3))+j
	rows[-2:]= np.array([0,N-1])
	columns[-2:]= np.array([0,N-1])
	
	data[-1], data[-2] = 1,1
	for i in range(1,N-1):
		data[3*(i-1)+2] = 1./(x[i+1]-x[i]) + c*1./2. + d*(x[i+1]-x[i])/6. # i, i+1 location
		data[3*(i-1)+1] = -( 1./(x[i+1]-x[i]) + 1./(x[i]-x[i-1]) 
							) + c*0. + d*(x[i+1]-x[i-1])/3. # i, i location
		data[3*(i-1)] = 1./(x[i]-x[i-1]) - c*1./2. + d*(x[i]-x[i-1])/6. # i, i-1 location
	A = coo_matrix((data, (rows,columns)), shape=(N,N))
	
	B = np.zeros(N)
	B[0], B[-1] = alpha, beta
	for j in range(1,N-1):
		B[j]= ( (x[j]-x[j-1])*func((x[j]+x[j-1])/2.)*(1./2.) + 
		    	(x[j+1]-x[j])*func((x[j+1]+x[j])/2.)*(1./2.)   )
		
	solution = spsolve(A.asformat('csr'),B)
	return x, solution

