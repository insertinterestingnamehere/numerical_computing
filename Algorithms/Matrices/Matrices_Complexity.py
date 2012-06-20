import scipy as sp
import scipy.linalg as la
import matplotlib.pyplot as plt

def Problem1():
    x = sp.linspace(-5, 5, 10)
    plt.plot(x, x*3, 'kD')
    plt.show()
    
def Problem2(x):
    x = sp.arange(x)
    return sp.array([x*i for i in xrange(x)])

def Problem3(x):
    numbers = sp.arange(x)
    return sp.outer(number, numbers)
    
def Problem4():
    #Need another problem
    
def Problem5():
    matrix = sp.zeros((10,10))
    matrix2 = sp.zeros((12,12))
    vector1 = sp.ones(12)
    
    try:
        print "Setting a array row length {0} with vector size 
        {1}".format(matrix[0].shape, vector1.shape)
        matrix[0] = vector1
    except ValueError, err:
        print "ValueError: ", err
        
    try:
        print "Concatenating a {0} size array with {1} size 
        array".format(matrix.shape, matrix2.shape)
        sp.concatenate((matrix, matrix2))
    except ValueError, err:
        print "ValueError: ", err    
    
def Problem6(h):
    x = sp.arange(0, sp.pi, h)
    approx = sp.diff(sp.sin(x**2))/h
    x = sp.delete(x, 0)
    actual = 2 * sp.cos(x**2) * x
    print "Error: ", sp.absolute(actual - approx).max()
    
    plt.plot(x, approx, x, actual, x, approx - actual)
    plt.show()
    
def Problem7():
    x = sp.rand(10000)
    print "Mean: {0} (0.5 - {0} = {1})".format(x.mean(), 0.5 - x.mean())
    print "Standard Deviation: {0} (1/sqrt(12) - {0} = 
    {1})".format(x.std(), 1./sp.math.sqrt(12) - x.std())
    
    
    
    
def Problem8(n):
    """Verify the numerical accuracy of linalg.lstsq vs la.inv"""
    from scipy.linalg import lstsq, inv, norm
    from scipy import dot, rand, allclose

    A = rand(n, n)
    b = rand(n, 1)

    inv_method = dot(inv(A), b)
    lstsq_method = lstsq(A, b)[0]

    #check the accuracy
    return norm(inv_method - lstsq_method)
