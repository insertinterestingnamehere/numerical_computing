from __future__ import division

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import spdiags, coo_matrix
from matplotlib import pyplot as plt

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
    # Number of finite elements is N-1
    # Number of basis functions is N
    
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



def Example3(N,epsilon):
    x, y = ode_fe(func=lambda x:-1./epsilon,c=-1./epsilon,x=np.linspace(0,1,N))
    plt.plot(x,y,'-ro')
    plt.axis([0.,1.1,.8,3.2])
    plt.show()
    return x, y


def NonlinearGrid():
    def asol(x,epsilon):
        A = []
        for item in x:
            A.append(alpha+item + (beta-alpha-1.)*(np.exp(item/epsilon) -1.)/(np.exp(1./epsilon) -1.)   )
        return x, np.array(A)
    
    epsilon = .01
    N=1000
    X = np.linspace(0,1,N)
    x1, y1 = ode_fe(func=lambda x:-1./epsilon,c=-1./epsilon,x=X)
    x2, y2 = ode_fe(func=lambda x:-1./epsilon,c=-1./epsilon,x=X**(1./14.))
    alpha, beta = 1.,3.
    # Analytic solution
    Z =asol(np.linspace(0,1,500),epsilon)
    
    plt.plot(x1,y1,'-bo')
    plt.plot(Z[0],Z[1],'-k',mfc="None")
    plt.plot(x2,y2,'-ro')
    plt.axis([0.,1.1,.8,3.2])
    plt.show()
    plt.clf()
    
    X = np.linspace(0,1,500)
    plt.plot(X,abs(Z[1]-interp1d(x2,y2)(X)),'-r')
    plt.plot( X,abs( Z[1]-interp1d(x1,y1)(X) ), '-b' )
    
    print "Max Error = ", np.max(np.abs(Z[1]-interp1d(x2,y2)(X) ))
    plt.show()



def cheb(N):
    def p(j1):
        if (j1==0 or j1 == N): return 2.
        else: return 1.
    
    x = np.cos(np.pi*np.arange(N+1)/N)
    D = np.zeros((N+1,N+1))
    # j represents column index
    for j in range(0,N+1):
        for i in range(0,j)+range(j+1,N+1):
            D[i,j] = ((-1.)**(i+j))*p(i)/( p(j)*(x[i]- x[j]) )
    
    # Values on the main diagonal
    for j in xrange(1,N): 
        D[j,j] = -x[j]/(2.*(1-x[j]**2.))
    D[0,0] = (1.+2.*N**2.)/6.
    D[N,N] = -(1.+2.*N**2.)/6.
    return D,x


def barycentric_interpolation_Chebychev(u,x,N,xx):
    uu = np.zeros(xx.shape)
    a = np.ones(N+1); a[1::2] = -1
    for x0 in range(len(xx)): 
        flag = False
        for item1,item2 in zip(x,u): 
            if abs(xx[x0] - item1)<1e-15:
                    flag = True
                    uu[x0] = item2
        if flag == False:
            b = xx[x0] - x
            b[0] *= 2.
            b[-1] *= 2.
            uu[x0] = np.sum((a*u)/b)/np.sum(a/b)
    return uu


class ODESolverSpectral:
    def __init__(self,N,order,lift,ode,dim=1):
        # N grid points, N-2 interior grid points, 
        # and N-1 subintervals, so self.N-1 will index 
        # the Nth grid point using Python indexing. 
        # The interior grid points will be indexed by 
        # 1:self.N-1
        self.N = N
        self.order = order
        self.lift = lift
        # ode must be a function with five arguments, 
        # u,x, y= list of differentiation matrices,
        # g = lifting function, and lmbda = parameter. 
        self.ode = ode
        self.dim = dim
        self.guess = np.zeros(self.dim*(self.N-2))
    
    def derivs(self):
        (D,x) = cheb(self.N-1)
        self.cheb_points = x
        self.operators = [[]]
        operator = D
        # print operator[1:self.N-1,1:self.N-1].shape
        # print self.cheb_points[1:self.N-1].shape
        for j in range(1,self.order+1):
            self.operators.append(operator[1:self.N-1,1:self.N-1])
            operator = operator.dot(D)
        # A_, D_, x_ = np.dot(D,D)[1:N-1,1:N-1], D[1:N-1,1:N-1], x[1:N-1]
    
    def ode_solve(self,output_grid):
        self.ode_func = (lambda u,x_=self.cheb_points[1:self.N-1],
                                        y=self.operators,G=self.lift: self.ode(u,x_,y,G)
                                        )
        sol = root(self.ode_func, self.guess )
        # self.sol = np.zeros((self.dim, len(self.cheb_points)))
        self.sol = np.zeros(self.cheb_points.shape)
        self.sol[1:self.N-1] = sol.x
        # print sol.x.shape, self.sol.shape
        # for j in range(self.dim):
        # 	print sol.x[0:self.N-2].shape
                # print self.dim, j
                # self.sol[j:1:self.N-1] = sol.x[(j)*(self.N-2):(j+1)*(self.N-2)]
        output_values = barycentric_interpolation_Chebychev(self.sol,
                                                                self.cheb_points,self.N-1,output_grid)
        return output_values
    
    def new_guess(self,new_guess):
        self.guess = new_guess
    


# Example3(N=100,epsilon=.01)
# NonlinearGrid()