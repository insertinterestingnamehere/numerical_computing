from cvxopt import matrix, solvers
##Q = 2*matrix([ [1., .5, 0.], [.5, 1., 0.], [0., 0., 1.] ])
##p = matrix([0.0, 0.0, 0.0])
##G = matrix([[-1.0,0.0],[0.0,-1.0],[0.,0.] ])
##h = matrix([0.0,0.0])
##A = matrix([1.0, 1.0, 0.], (1,3))
##b = matrix(1.0)
##sol=solvers.qp(Q, p, G, h, A, b)
##print(sol['x'])

##Q = matrix([ [4., 2.], [2., 2.] ])
##p = matrix([1., -1.])
##G = matrix([[1., 0.], [0., -1.]])
##h = matrix([0.,0.])
##sol=solvers.qp(Q, p, G, h)
##print(sol['x'])
##print sol['primal objective']

##Q = matrix([ [3., 0., 1.], [0., 4., 2.], [1., 2., 3.] ])
##p = matrix([-3.0, 0.0, -1.0])
##sol=solvers.qp(Q, p)
##print(sol['x'])
##print sol['primal objective']

def g(x):
    a = x[0] - 4
    b = x[1] - 3
    c = x[2] + 5
    return a**4 + b**2 + 4*(c**4)

m = matrix([0,0,0])
F = [0,m]

x = [0,0,1]
print g(x)

sol = solvers.cp(g(x)
                 )
