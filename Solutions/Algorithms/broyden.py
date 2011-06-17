def broyden1d(func, xpts, tol=0.00005, iter=30):
    """Find the zero of a function between two points (accepted as a list)

    When f(pt2) < tol, we are close enough to a zero and stop"""

    #truncate list to two item and sort
    x = xpts[:2]
    x.sort()

    #check our original points for zeros
    if abs(func(x[0])) < tol:
        return x[0]
    elif abs(func(x[1])) < tol:
        return x[1]

    #calculate our second guess
    x_1, x_2 = float(x[0]), float(x[1])
    fx1 = func(x_1)
    for i in xrange(iter):
        x_new = x_1-fx1*((x_1-x_2)/(fx1-func(x_2)))

        if abs(func(x_new)) < tol:
            return x_new
        else:
            x_2 = x_new
    return ValueError("No Zeros found in %d iterations" % iter)


def regula_falsi(func, xpts, tol=0.00005, iter=30):
    """Find the zero of a function between two points (accepted as a list)

    This method uses the regula falsi secant/bisection method to converge on the
    root."""

    #truncate list to two tiems and sort
    x=xpts[:2]
    x.sort()

    #check that func(x[0])<0 and func(x[1])>0
    if not(func(x[0])<0 and func(x[1])>0):
        raise ValueError


    #check our original points for zeros
    if abs(func(x[0])) < tol:
        return x[0]
    elif abs(func(x[1])) < tol:
        return x[1]

    #calculate our second guess
    x_1, x_2= float(x[0]), float(x[1])
    x_old = x_2
    fx1 = func(x_1)
    for i in xrange(iter):
        x_new = x_1-fx1*((x_1-x_2)/(fx1-func(x_2)))

        if abs(x_new-x_old) < tol:
            return x_new
        else:
            x_old = x_new
            tmp = func(x_new)
            if x_1*x_2 > 0.0:
                x_1 = x_new
            else:
                x_2 = x_new
    return ValueError("No Zeros found in %d iterations" % iter)

#Read FiniteDiff labs.
def broyden(func, xpts, tol=0.00005, iter=30):
    """Calculate the zero of a multi-dimensional function using Broyden's method"""

    #truncate list to two tiems and sort
    x=xpts[:2]
    x.sort()

    #check our original points for zeros
    if abs(func(x[0])) < tol:
        return x[0]
    elif abs(func(x[1])) < tol:
        return x[1]

    x_1, x_2 = float(x[0]), float(x[1])

    def jacobian(funcs, vars):
        return [[sp.diff(f, v) for v in vars] for f in funcs]

    Fgrad = func(x
    J_new = J_old+(
