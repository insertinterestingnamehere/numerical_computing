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

def gaussNewton(f, Df, Jac, r, x, niter=10):
    '''
    Solve a nonlinear least squares problem with Gauss-Newton method.
    Inputs:
        f -- the objective function
        Df -- gradient of f
        Jac -- jacobian of residual vector
        r -- the residual vector
        x -- initial point
        niter -- integer giving the number of iterations
    Returns:
        the minimizer
    '''
    a=0
    for i in xrange(niter):
        J = Jac(x)
        g = J.T.dot(r(x))
        p = la.solve(J.T.dot(J), -g)
        slope = (g*p).sum()
        a = opt.line_search(f, Df, x, p)[0]  
        x += a*p
    return x

#US census data example
years1 = np.arange(8)
pop1 = np.array([3.929, 5.308, 7.240, 9.638, 12.866, 17.069, 23.192, 31.443])
years2 = np.arange(16)
pop2 = np.array([3.929, 5.308, 7.240, 9.638, 12.866, 17.069, 23.192, 31.443,
                 38.558, 50.156, 62.948, 75.996, 91.972, 105.711, 122.775, 131.669])
def model1(x, t):
    return x[0]*np.exp(x[1]*(t+x[2]))
def residual1(x):
    return model1(x, years1) - pop1
guess1 = np.array([1., .4, 2.5])
x1 = opt.leastsq(residual1, guess1)[0]

def model2(x,t):
    return x[0]/(1+np.exp(-x[1]*(t+x[2])))
def residual2(x):
    return model2(x, years2) - pop2
guess2 = np.array([150., .4, -15.])
x2 = opt.leastsq(residual2, guess2)[0]

guess1 = np.array([150., .4, 2.5])
dom = np.linspace(0,8,100)
plt.plot(dom, x1[0]*np.exp(x1[1]*(dom+x1[2])))
plt.scatter(years1,pop1)
plt.show()

guess2 = np.array([150., .4, -15.])
dom = np.linspace(0,16,100)
plt.plot(dom, model2(x2,dom))
plt.scatter(years2,pop2)
plt.show()
