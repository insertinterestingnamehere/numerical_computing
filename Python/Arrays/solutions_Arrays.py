import math
import timeit
import numpy as np
from numpy.random import randn
from scipy.misc import factorial
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#problem 1

def arrmul(A,B):
    new = []
    for i in range(len(A)):
        newrow = []
        for k in range(len(B[0])):
            tot = 0
            for j in range(len(B)):
                tot += A[i][j] * B[j][k]
            newrow.append(tot)
        new.append(newrow)
    return new
    
def timefunction(f, *args, **kwargs):
	pfunc = lambda: f(*args, **kwargs)
	print min(timeit.repeat(pfunc, number = 1, repeat = 1))

k = 100
A = [range(i, i+k) for i in range(0, k**2, k)]
B = [range(i, i+k) for i in range (0, k**2, k)]

#timefunction(numpy.dot, NumA, NumA
#timefunction(arrmul, A, B)
'''
Lists
k=100: 0.195740438121
k=200: 1.96796994247
k=300: 7.87688692047

Arrays
k=100: 0.000890023231932
k=200: 0.00714212847242
k=300: 0.0233234591569

It takes significantly less time to square a two dimensional NumPy array 
than it does to square a two dimensional list. This is because Python is a 
high level interpreted language and thus slower than lower level compiled 
languages. NumPy has heavily optimized algorithms that use Python to run code that 
has been written and optimized in other languages (usually C or Fortran)
'''

#problem 2
def problem2():
    A = rand(1000,1000)
    #timeit A.reshape(A.size)
    #timeit A.flatten()
    #timeit A.reshape((1, A.size))
    print "A.reshape(A.size) had a best time of 5.65 microseconds"
    print "A.flatten() had a best time of 26.2 milliseconds"
    print "A.reshape((1, A.size)) had a best time of 3.15 microseconds"
    '''
    A.flatten() takes longer because it is allocating a new array 
    in memory and copying all of the values from the input array into
    the new array. Will return a copy (which takes more time).
    
    A.reshape() only changes the way the array is read from memory by changing
    the shape of the array. It doesn't touch any of the data of the array. 
    Will return a view (which takes less time) if possible. 
    
  	'''

# problem 3
def laplace(U, tol):
    new = U.copy()
    dif = tol
    while tol <= dif:
        new[1:-1,1:-1] = (U[:-2,1:-1] + U[2:,1:-1] + U[1:-1,:-2] + U[1:-1,2:])/4.0
        dif = np.max(np.absolute(U-new))
        U[:] = new
        
n = 100
tol = .0001	
U = np.ones ((n , n ))
U [:,0] = 100 # set north boundary condition
U [:,-1] = 100 # set south boundary condition
U [0] = 0 # set west boundary condition
U [-1] = 0 # set east boundary condition
laplace(U, tol) # U has been changed in place
x = np.linspace (0, 1, n)
y = np.linspace (0, 1, n)
X, Y = np.meshgrid (x, y)
fig = plt.figure()
ax = fig.gca( projection = '3d')
ax.plot_surface (X, Y, U, rstride=5)
plt.show()


# problem 4

#as n increases the variance approaches 0.

def large_numbers(n):
	# demonstrates law of large numbers
	# as n increases, variance goes to 0.
    A = randn(n, n)
    return A.mean(axis=1).var()
    
    
# problem 5
def rgb():
	A = np.random.randint(0, 256, (100, 100, 3))
	A * [.5, .5, 1]

# problem 6 
def arcsin_approx():
	n = 70
	s = 1. * np.arange(70,-1,-1)
	r = factorial(2*s)/((2*s+1)*(factorial(s)**2)*(4**s)) # computes coefficients
	q = np.zeros(142)
	q[0::2] = r
	P = np.poly1d(q)
	return P(1/math.sqrt(2))*4

def W_approx():
	n = 20
	s = 1. * np.arange(20,0,-1)
	r = ((-s)**(s-1))/(factorial(s)) # computes coefficients
	q = np.zeros(21)
	q[0:-1] = r
	P = np.poly1d(q)
	return P(.25)
	
print W_approx()*math.e**W_approx() #verification! 
    
    
    

# The problems from here on are no longer in the first lab.

# problem 6
def egcd(a, b):
    '''
    Extended Euclidean algorithm
    Returns (b, x, y) such that mx + ny = b
    Source: http://en.wikibooks.org/wiki/Algorithm_Implementation/Mathematics/Extended_Euclidean_algorithm
    '''
    x,y, u,v = 0,1, 1,0
    while a != 0:
        q,r = b//a,b%a; m,n = x-u*q,y-v*q
        b,a, x,y, u,v = a,r, u,v, m,n
    return b, x, y

def modinv(a, m):
    '''
    Find the modular inverse.
    Source: http://en.wikibooks.org/wiki/Algorithm_Implementation/Mathematics/Extended_Euclidean_algorithm
    '''
    g, x, y = egcd(a, m)
    if g != 1:
        return None  # modular inverse does not exist
    else:
        return x % m

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.izip_longest(fillvalue=fillvalue, *args)

def blockize(msg, n):
    lut = {a:i for i ,a in enumerate(string.lowercase)}
    msg = "".join(msg.lower().split())
    return list(map(np.array, grouper(map(lut.__getitem__, msg), n, fillvalue=lut['x'])))

def inv_mat(n):
    tries = 0
    while True:
        a = np.random.randint(1000, size=(n, n)) % 26
        d = round(linalg.det(a))
        
        if gcd(int(d), 26) == 1:
            break
        tries += 1
            
    return a, d

def encode(msg, k):
    ciphertext = []
    n = k.shape[0]
    ilut = {i:a for i, a in enumerate(string.lowercase)}
    for i in blockize(msg, n):
        s = i.dot(k) % 26
        ciphertext.append("".join(map(ilut.__getitem__, s)))
    
    return "".join(ciphertext)


def inv_key(key):
    d = round(linalg.det(key))
    inv_d = modinv(int(d), 26)
    ik = np.round(d*linalg.inv(key))
    return (ik*inv_d) % 26
    
def decode(msg, k):
    ik = inv_key(k)
    n = ik.shape[0]
    plaintext = []
    ilut = {i:a for i, a in enumerate(string.lowercase)}
    for i in blockize(msg, n):
        s = i.dot(ik) % 26
        plaintext.append("".join(map(ilut.__getitem__, s)))
        
    return "".join(plaintext)
    
    
def prob5():
    im = np.random.randint(1,256,(100,100,3))
    b = np.array([0.5,0.5,1])
    im_bluer = (im * b).astype(int)

def broadcast_1():
    """All input arrays have exactly the same shape"""
    a = np.random.rand(4, 5)
    b = np.random.rand(4, 5)
    r = a * b
    print "Case 1: {} * {} = {}".format(a.shape, b.shape, r.shape)

def broadcast_2():
    """All input arrays are of the same dimension and
    the length of corresponding dimensions match or is 1"""

    a = np.random.rand(5, 4, 1, 6)
    b = np.random.rand(5, 4, 1, 1)
    r = a * b
    print "Case 2: {} * {} = {}".format(a.shape, b.shape, r.shape)

def broadcast_3():
    """All input arrays of fewer dimension can have 1
    prepended to their shapes to satisfy the second criteria."""

    a = np.random.rand(1, 6)
    b = np.random.rand(5, 4, 1, 6)
    r = a * b
    print "Case 3: {} * {} = {}".format(a.shape, b.shape, r.shape)

def series_problem_a():
	c = np.arange(70, -1, -1) # original values for n
	c = factorial(2*c) / ((2*c+1) * factorial(c)**2 * 4**c) #series coeff's
	p = np.zeros(2*c.size) # make space for skipped zero-terms
	p[::2] = c # set nonzero polynomial terms to the series coeff's
	P = np.poly1d(p) # make a polynomial out of it
	return 6 * P(.5) #return pi (since pi/6 = arcsin(1/2))

def series_problem_b():
	p = np.arange(20, -1, -1) # original values for n
	p = (-p)**(p-1) / factorial(p) #compute coefficients
	p[-1] = 0. # get rid of NAN in the zero-term
	P = np.poly1d(p) # Make a polynomial
	print P(.25) * np.exp(P(.25)) # test it
	return P(.25) # return the computed value


