import scipy as sp
import FiniteDiff as FD

def der(fc, x, h=.0001, degree=1, type='centered', accuracy=2):
    """ Computes the numerical of the callable function 'fc at all the
    points in array 'x'. 'degree' is the degree of the derivative to be
    computed. 'type' can be 'centered', 'forward', or 'backward'.
    'accuracy' is the desired order of accuracy. For forward and backward
    differences it can take a value of 1, 2, or 3. For centered differences
    it can take a value of 2, 4, or 6."""
    # Use these lists to manage the different coefficient options.
    A = np.array([[[0., 0., -.5, 0., .5, 0., 0.],
                   [0., 1/12., -2/3., 0., 2/3., -1/12., 0.],
                   [-1/60., 3/20., -3/4., 0., 3/4., -3/20., 1/60.]],
                  [[0., 0., 1., -2., 1., 0., 0.],
                   [0., -1/12., 4/3., -5/2., 4/3., -1/12., 0.],
                   [1/90., -3/20., 3/2., -49/18., 3/2., -3/20., 1/90.]]])
    B = np.array([[[-1., 1., 0., 0., 0.],
                   [-1.5, 2., -.5, 0., 0.],
                   [-11/6., 3., -1.5, 1/3., 0.]],
                  [[1., -2., 1., 0., 0.],
                   [2., -5., 4., -1., 0.],
                   [35/12., -26/3., 19/2., -14/3., 11/12.]]])
    if type == "centered":
        acc = int(accuracy/2) - 1
    else:
        acc = int(accuracy) - 1
    if int(degree) not in [1, 2]:
        raise ValueError ("Only first and second derivatives are supported")
    if acc not in [0, 1, 2]:
        raise ValueError ("Invalid accuracy")
    if type == 'centered':
        xdifs = np.array([fc(x+i*h) for i in xrange(-3, 4)])
        return np.inner(A[degree-1,acc], xdifs.T) / h**degree
    elif type == 'forward':
        xdifs = np.array([fc(x+i*h) for i in xrange(5)])
        return np.inner(B[degree-1,acc], xdifs.T) / h**degree
    elif type == 'backward':
        xdifs = np.array([fc(x-i*h) for i in xrange(5)])
        return np.inner(B[degree-1,acc], xdifs.T) / (-h)**degree
    else:
        raise ValueError ("invalid type")

def partial(fc, x, i, h=.0001, ty="centered", ac=2):
    """ Computes a partial derivative with respect to index 'i'.
    The rest of the options are the same as the numerical derivative function."""
    def fcpart(y):
        add = np.zeros(x.shape[0])
        add[i] = y
        return fc(x+add)
    return der(fcpart, 0., h=h, type=ty, accuracy=ac)

def jac(fc, x, ty="centered", ac=2, h=.0001):
    """Compute the Jacobian matrix of a function.
    'fc' is a callable function that operates on a 1D array.
    'x' is where to evaluate the Jacobian matrix.
    Dimensions of the domain and range are infered from 'x'
    and the output of 'fc'."""
    return np.array([partial(fc, x, [i], h=h, ty=ty, ac=ac) for i in xrange(x.size)]).T

def multipartial(fc, x, li, h=.0001, ty="centered", ac=2):
    """ Computes multiple partial derivatives via recursion.
    'fc' is a callable function. 'x' is where to take the derivatives.
    'li' is a list of indices corresponding to the partials to be taken."""
    if len(li) <= 1:
        return partial(fc, x, li[0], h=h, ty=ty, ac=ac)
    else:
        part = lambda x: partial(fc, x, li[-1], h=h, ty=ty, ac=ac)
        return multipartial(part, x, li[:-1], h=h, ty=ty, ac=ac)

def hessian(fc, x, h=.0001, ty="centered", ac=2):
    """ Hessian matrix of function 'fc' at point 'x'.
    Computed using difference 'h' and difference type 'ty' of acccuracy 'ac'.
    Exact options are the same as the numerical derivative function."""
    hes = np.empty((x.size, x.size))
    for i in xrange(x.size):
        for j in xrange(x.size):
            hes[i,j] = multipartial(fc, x, [i,j], h=h, ty=ty, ac=ac)
    return hes

# Here are alternate versions for the Jacobian and Hessian functions.
def Jacobian(f, m, n, pt, mode='centered', o=2, h=1e-5):
    '''
    Approximate the Jacobian of a function at a point.
    Inputs:
        f -- a callable function whose jacobian we will approximate
        m -- integer giving the dimension of the domain or f
        n -- integer giving the dimension of the range of f
        pt -- numpy array specifying the point at which to approximate the Jacobian
        mode -- specifies the type of difference scheme. Should take values in
                ['centered', 'backward', 'forward'].
        d -- the order of the derivative. Should take values in [1,2]
        o -- order of approximation. If mode = 'centered', should take values
             [2,4,6], otherwise should take values in [1,2,3]
        h -- the size of the difference step
    Returns:
        jac -- array the same shape as pts, giving the approximate Jacobian at 
               each point in pts.
    '''
    jac = np.empty((m,n))
    for i in xrange(n):
        off = np.zeros(n)
        off[i] = h
        if mode == 'centered':
            if o == 2:
                jac[:, i] = (-.5*f(pt - off) + .5*f(pt + off))/h
            if o == 4:
                jac[:, i] = (f(pt-2*off)/12 - 2*f(pt-off)/3 + 2*f(pt+off)/3 - f(pt+2*off)/12)/h
            if o == 6:
                jac[:, i] = (-f(pt-3*off)/60 + 3*f(pt-2*off)/20 - 3*f(pt-off)/4 + 
                          f(pt+3*off)/60 - 3*f(pt+2*off)/20 + 3*f(pt+off)/4)/h
        if mode == 'forward':
            if o == 1:
                jac[:, i] = (-f(pt)+f(pt+off))/h
            if o == 2:
                jac[:, i] = (-3*f(pt)/2 + 2*f(pt+off) - f(pt+2*off)/2)/h
            if o == 3:
                jac[:, i] = (-11*f(pt)/6 + 3*f(pt+off) - 3*f(pt+2*off)/2 + f(pt+3*off)/3)/h
        if mode == 'backward':
            if o == 1:
                jac[:, i] = -(-f(pt)+f(pt-off))/h
            if o == 2:
                jac[:, i] = -(-3*f(pt)/2 + 2*f(pt-off) - f(pt-2*off)/2)/h
            if o == 3:
                jac[:, i] = -(-11*f(pt)/6 + 3*f(pt-off) - 3*f(pt-2*off)/2 + f(pt-3*off)/3)/h
    return jac

def Hessian(f, n, pt, h=1e-5):
    '''
    Approximate the Hessian of a function at a given point.
    Inputs:
        f -- callable function object
        n -- integer giving the dimension of the domain of f
        pt -- flat numpy array giving the point at which to approximate
        h -- step size for difference scheme
    Returns:
        hess -- numpy array giving the approximated Hessian matrix
    '''
    hess = np.empty((n,n))
    off = np.eye(n)*h
    for i in xrange(n):
        for j in xrange(n):
            hess[i,j] = (f(pt+off[i]+off[j])-f(pt + off[i]-off[j]) -
                         f(x+off[j]-off[i]) + f(pt-off[i]-off[j]))/(4*h**2)
    return hess
