import numpy as np
from scipy.optimize import fmin_cg


def conjugateGradient(b, x, mult):
    '''
    Minimize .5x^TQx - b^Tx + c using the method of Steepest Descent.
    Equivalently, solve Qx = b.
    Inputs:
        b -- length n array
        x -- length n array, the inital guess
        mult -- a callable function object that performs matrix-vector multiplication by Q.
                i.e., mult(d) returns Qd.
    Returns:
        a numpy array, the solution to Qx = b.
    '''
    r = mult(x) - b
    d = -r.copy()
    n2 = (r*r).sum()
    while not np.allclose((r*r).sum(), 0):
        n1 = n2
        m = mult(d)
        a = (r*r).sum()/(d*m).sum()
        x += a*d
        rnew = r + a*m
        n2 = (rnew*rnew).sum()
        beta = n2/n1
        d = -rnew + beta * d
        r = rnew
    return x

    
def linRegression():
    '''http://www.itl.nist.gov/div898/strd/lls/data/LINKS/v-Longley.shtml'''
    
    def mult(x):
        return Q.dot(x)
    
    data =np.loadtxt('linregression.txt')
    y = data[:,0]
    A = np.ones(data.shape)
    A[:,1:] = data[:,1:]
    Q = A.T.dot(A)
    b = A.T.dot(y)
    x0 = np.random.random(Q.shape[1])
    x = conjugateGradient(b, x0, mult)
    print x


def logRegression():
    '''This data is simulated'''
    
    def objective(b):
        '''Return -1*l(b), where l is the log likelihood.'''
        return np.log(1+np.exp(np.dot(x,b))).sum() - (y*(np.dot(x,b))).sum()
    
    data = np.loadtxt('logregression.txt')
    y = data[:,0]
    x = np.ones(data.shape)
    x[:,1:] = data[:,1:]
    guess = np.array([1., 1., 1., 1.])
    fmin_cg(objective, guess)
