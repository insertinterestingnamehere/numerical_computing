import scipy as sp
import scipy.linalg as la


'''
The Newton's lab in applications needed an overloaded newtons method function
When I found that python didn't have function overloading, this is what I came
up with. I don't really like it though
'''
def newtonsMethod(f, x0, tol=1e-7, df=None):
	if df is None:
        df = lambda x: sp.derivative(f,x)
    
    x = x0
	while(sp.absolute(float(f(x))/df(x)) >= tol):
		x -= float(f(x))/df(x)
	return x

'''
Tested it like this
%timeit newtonsMethod(lambda x: sp.sin(x),lambda x: sp.cos(x),0,0.001)
%timeit newtonsMethod(lambda x: sp.sin(x),0,0.001)


When I tested it out in python, hand fed derivatives performed  faster
cos: 10 times as fast
sin(1/x)x^2: 7 times as fast
sin(x)/x - x: 5 times as fast
sin(x)/x + x: 6 times as fast
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
