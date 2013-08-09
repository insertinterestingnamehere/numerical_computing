import scipy as sp
import scipy.linalg as la
import numpy.linalg as npla

#Problem 1
A = sp.array([[.75,.50],[.25,.50]])
npla.matrix_power(A,2)[0,0]#Part a: 0.6875
npla.matrix_power(A,100)[0,0]#Part b: 66.7%

#Problem 2
A = sp.array([[1./4,1./3,1./2],[1./4,1./3,1./3],[1./2,1./3,1./6]]);A #Part a
npla.matrix_power(A,2)[1,0]#Part b: 0.3125
V = la.eig(A)[1]
x = V[:,0]
la.eig(A)[1]
x = x/sp.sum(x);x #Part c: array([ 0.35955056,  0.30337079,  0.33707865])

#Problem 3
bucky = sp.loadtxt ( "bucky.csv" , delimiter = "," )

sp.count_nonzero(bucky)
sp.count_nonzero(npla.matrix_power(bucky,2))
sp.count_nonzero(npla.matrix_power(bucky,3))

sp.count_nonzero(npla.matrix_power(bucky,9)) #3600, all atoms are connected, so path length 9


