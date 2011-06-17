import scipy as sp
import scipy.linalg as la
import matplotlib.pyplot as plt

#Problem 1
x = sp.linspace(-5,5,10)
plt.plot(x,3*x,'kD')
plt.show()

#Problem 2
x = sp.arange(1,7)
sp.array([x,2*x,3*x,4*x,5*x,6*x])

#Problem 3
sp.vstack(sp.arange(1,7))*sp.arange(1,7)

#Problem 4
bucky = sp.loadtxt ( "bucky.csv" , delimiter = "," )
bucky.size

#Problem 5
#???? Following the directions, I failed to get the desired errors -Forrest

#Problem 6
h = .001
x = sp.arange(0,sp.pi,h)
approx = sp.diff(sp.sin(x**2))/h
x = sp.delete(x,0)
actual = 2*sp.cos(x**2)*x

sp.absolute(actual-approx).max()#Absolute Max Difference, the error

plt.plot(x,approx,x,actual,x,approx-actual)
plt.show()

#Problem 7
x = sp.rand(10000)
x.mean()#(0+1)/2 = .5 -- Its close
.5 - x.mean()
x.std()#(1-0)/sqrt(12) = 1/sqrt(12) = .288675 also close
1/sp.sqrt(12)-x.std()

#Problem 8

def leastSquares(A,b):
	return sp.dot( sp.dot( la.inv( sp.dot(A.T,A) ),A.T ) ,b )
	
h = 450
w = 400

A = sp.rand(h,w);
b = sp.rand(h,1);

x1 = leastSquares(A,b)
x2 = la.lstsq(A,b)[0]

print(la.norm(x1-x2))
