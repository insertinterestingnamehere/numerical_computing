'''
Solutions file for the line search lab.
'''

def newtons1d(f, df, ddf, x, niter=10):
    '''
    Perform Newton's method to minimize a function from R to R.
    Inputs:
        f -- objective function (twice differentiable)
        df -- first derivative
        ddf -- second derivative
        x -- initial guess
        niter -- integer, giving the number of iterations
    Returns:
        the approximated minimizer
    '''
    for i in xrange(niter):
        x = x-df(x)/ddf(x)
    return x
    
def backtracking(f, slope, x, p, a=1, rho=.9, c=10e-4):
    '''
    Perform a backtracking line search to satisfy the Wolfe Conditions.
    Return the step length.
    Inputs:
        f -- the objective function
        slope -- equal to grad(f)^T p
        x -- current iterate
        p -- current direction
        a -- intial step length (set to 1 in Newton and quasi-Newton methods)
        rho -- number in (0,1)
        c -- number in (0,1)
    Returns:
        the computed step size
    '''
    b = f(x)
    while f(x+a*p) > b + c*a*slope:
        a = rho*a
    return a
    
def gradientDescent(f, Df, x, niter=10):
    '''
    Minimize a function using gradient descent.
    Inputs:
        f -- differentiable real-valued function
        Df -- the gradient of the function
        x -- initial point
        niter -- integer giving the number of iterations to run.
    Returns:
        a list, the sequence of points generated
    '''
    pts = []
    pts.append(x.copy())
    for i in xrange(niter):
        p = -Df(x)
        slope = (p**2).sum()
        a = backtracking(f, slope, x, p)
        x += a*p
        pts.append(x.copy())
    return pts

def newtonsMethod(f, Df, DDf, x, niter=10):
    '''
    Minimize a function using Newton's method.
    Inputs:
        f -- real-valued, twice-differentiable function
        Df -- the gradient of the function
        DDf -- the Hessian of the function
        x -- initial point
        niter -- integer giving the number of iterations
    Returns:
        a list, the sequence of points generated
    '''
    pts = [x.copy()]
    for i in xrange(niter):
        p = la.solve(DDf(x),-Df(x))
        slope = (p**2).sum()
        a = backtracking(f, slope, x, p)
        x += a*p
        pts.append(x.copy())
    return pts
