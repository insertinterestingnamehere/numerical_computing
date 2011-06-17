import scipy as sp
import scipy.linalg as la


'''
The Newton's lab in applications needed an overloaded newtons method function
When I found that python didn't have function overloading, this is what I came
up with. I don't really like it though
'''
def newtonsMethod( *args):
	f = None
	df = None
	x0 = None
	tol = None
	if ( len(args) == 3 ):
		f = args[0]
		df = lambda x: sp.derivative(f,x)
		x0 = args[1]
		tol = args[2]
	else :
		f = args[0]
		df = args[1]
		x0 = args[2]
		tol = args[3]
		
	x = x0
	while( sp.absolute(float(f(x))/df(x)) >= tol):
		x -= float(f(x))/df(x)
	return x

if __name__ == "__main__":
	from timer import timer

	with timer(loops=10) as t:
		print("\nsp.cos(x)")
		print('-'*20)
		print(newtonsMethod(lambda x: sp.cos(x),0.1,0.001))
		t.time(newtonsMethod,lambda x: sp.cos(x),0.1,0.001)
		t.printTime()
		print("With Derivative")
		print(newtonsMethod(lambda x: sp.cos(x),lambda x: -sp.sin(x),0.1,0.001))
		t.time(newtonsMethod,lambda x: sp.cos(x),lambda x: -sp.sin(x),0.1,0.001)
		t.printTime()
		
		print("\nsp.sin(1/x)*x**2")
		print('-'*20)
		print(newtonsMethod(lambda x: sp.sin(1/x)*x**2,0.1,0.001))
		t.time(newtonsMethod,lambda x: sp.sin(1/x)*x**2,0.1,0.001)
		t.printTime()
		print("With Derivative")
		print(newtonsMethod(lambda x: sp.sin(1/x)*x**2,lambda x: -sp.cos(1/x) + 2*x*sp.sin(1/x),0.1,0.001))
		t.time(newtonsMethod,lambda x: sp.sin(1/x)*x**2,lambda x: -sp.cos(1/x) + 2*x*sp.sin(1/x),0.1,0.001)
		t.printTime()
		
		print("\nsp.sin(x)/x -x")
		print('-'*20)
		print(newtonsMethod(lambda x: sp.sin(x)/x -x,0.1,0.001))
		t.time(newtonsMethod,lambda x: sp.sin(x)/x -x,0.1,0.001)
		t.printTime()
		print("With Derivative")
		print(newtonsMethod(lambda x: sp.sin(x)/x -x,lambda x: sp.cos(x)/x -sp.sin(x)/x**2 -1,0.1,0.001))
		t.time(newtonsMethod,lambda x: sp.sin(x)/x -x,lambda x: sp.cos(x)/x -sp.sin(x)/x**2 -1,0.1,0.001)
		t.printTime()
	
		print("\nsp.sin(x)/x +x")
		print('-'*20)
		print(newtonsMethod(lambda x: sp.sin(x)/x +x,0.1,0.001))
		t.time(newtonsMethod,lambda x: sp.sin(x)/x +x,0.1,0.001)
		t.printTime()
		print("With Derivative")
		print(newtonsMethod(lambda x: sp.sin(x)/x +x,lambda x: sp.cos(x)/x -sp.sin(x)/x**2 +1,0.1,0.001))
		t.time(newtonsMethod,lambda x: sp.sin(x)/x +x,lambda x: sp.cos(x)/x -sp.sin(x)/x**2 +1,0.1,0.001)
		t.printTime()
	'''
	We should specify error bounds and initial guesses for roots
	for some I had trouble with starting guesses producing a zero-valued derivative
	'''


#Problem 4
def newtonsMatrix( *args):

	def jacobian( F,x):
	
		def replace( A,a,i):
			R=A.copy() #This line caused me a lot of problems
			R[i]=a
			return R
			
		J = sp.zeros((len(x),len(x)))
		for i in range(len(x)):
			for j in range(len(x)):
				#Is there a better way to do a partial derivative?
				J[i,j] = sp.derivative(lambda a: F(replace(x,a,i))[j],x[i])
		return J
		
	F = None
	J = None
	x0 = None
	tol = None
	if ( len(args) == 3 ):
		F = args[0]
		J = lambda x: jacobian(F,x)
		x0 = args[1]
		tol = args[2]
	else :
		F = args[0]
		J = args[1]
		x0 = args[2]
		tol = args[3]
		
	x = x0
	inc = la.solve(J(x),F(x))
	while( sp.absolute(inc).max() >= tol):
		x -= inc
		inc = la.solve( J(x),F(x))
	return x
	
if __name__ == "__main__":
	from timer import timer

	with timer(loops=10) as t:
		print("\n x**2 + y**2 = 1 and (x-1)**2 + (y-1)**2 = 1")
		F = lambda x: sp.array([[x[0,0]**2 +x[1,0]**2 -1],[(x[0,0]-1)**2+(x[1,0]-1)**2 -1]])
		J = lambda x:sp.atleast_2d(sp.array([ [ 2*x[0,0],2*(x[0,0]-1)],[2*x[1,0],2*(x[1,0]-1)]]))
		x0 = sp.array([[.75],[.5]])
		
		a = newtonsMatrix(F,x0,0.00001)
		print(a)
		print(F(a))
		t.time(newtonsMatrix,F,x0,0.00001)
		t.printTime()
		
		print("With Jacobian")
		a = newtonsMatrix(F,J,x0,0.00001)
		print(a)
		print(F(a))
		t.time(newtonsMatrix,F,J,x0,0.00001)
		t.printTime()
		
		
		

