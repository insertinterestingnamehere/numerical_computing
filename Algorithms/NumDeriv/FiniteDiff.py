import numpy as np

def numDer(f, pts, mode='centered', d=1, o=2, h=1e-5):
    '''
    Approximate the derivative of a function at an array of points.
    Inputs:
        f -- a callable function whose derivative we will approximate
        pts -- numpy array of points at which to approximate the derivative
        mode -- specifies the type of difference scheme. Should take values in
                ['centered', 'backward', 'forward'].
        d -- the order of the derivative. Should take values in [1,2]
        o -- order of approximation. If mode = 'centered', should take values
             [2,4,6], otherwise should take values in [1,2,3]
        h -- the size of the difference step
    Returns:
        app -- array the same shape as pts, giving the approximate derivative at 
               each point in pts.
    '''
    # this implementation has the difference coefficients hard-coded in.
    # it is fairly straight-forward, contains a bit of code duplication
    # initialize an array that will hold the approximations
    app = np.empty(pts.shape)
    if mode == 'centered':
        if d == 1:
            if o == 2:
                app[:] = (-.5*f(pts - h) + .5*f(pts + h))/h
            if o == 4:
                app[:] = (f(pts-2*h)/12 - 2*f(pts-h)/3 + 2*f(pts+h)/3 - f(pts+2*h)/12)/h
            if o == 6:
                app[:] = (-f(pts-3*h)/60 + 3*f(pts-2*h)/20 - 3*f(pts-h)/4 + 
                          f(pts+3*h)/60 - 3*f(pts+2*h)/20 + 3*f(pts+h)/4)/h
        if d == 2:
            if o == 2:
                app[:] = (f(pts - h) - 2*f(pts) + f(pts + h))/h**2
            if o == 4:
                app[:] = (-f(pts-2*h)/12 + 4*f(pts-h)/3 - 5*f(pts)/2 + 4*f(pts+h)/3 
                          - f(pts+2*h)/12)/h**2
            if o == 6:
                app[:] = (f(pts-3*h)/90 - 3*f(pts-2*h)/20 + 3*f(pts-h)/2 - 49*f(pts)/18 + 
                          f(pts+3*h)/90 - 3*f(pts+2*h)/20 + 3*f(pts+h)/2)/h**2
    if mode == 'forward':
        if d==1:
            if o == 1:
                app[:] = (-f(pts)+f(pts+h))/h
            if o == 2:
                app[:] = (-3*f(pts)/2 + 2*f(pts+h) - f(pts+2*h)/2)/h
            if o == 3:
                app[:] = (-11*f(pts)/6 + 3*f(pts+h) - 3*f(pts+2*h)/2 + f(pts+3*h)/3)/h
        if d == 2:
            if o == 1:
                app[:] = (f(pts) - 2*f(pts+h) + f(pts+2*h))/h**2
            if o == 2:
                app[:] = (2*f(pts) - 5*f(pts+h) + 4*f(pts+2*h) - f(pts+3*h))/h**2
            if o == 3:
                app[:] = (35*f(pts)/12 - 26*f(pts+h)/3 + 19*f(pts+2*h)/2 - 
                          14*f(pts+3*h)/3 + 11*f(pts+4*h)/12)/h**2
    if mode == 'backward':
        if d==1:
            if o == 1:
                app[:] = -(-f(pts)+f(pts-h))/h
            if o == 2:
                app[:] = -(-3*f(pts)/2 + 2*f(pts-h) - f(pts-2*h)/2)/h
            if o == 3:
                app[:] = -(-11*f(pts)/6 + 3*f(pts-h) - 3*f(pts-2*h)/2 + f(pts-3*h)/3)/h
        if d == 2:
            if o == 1:
                app[:] = (f(pts) - 2*f(pts-h) + f(pts-2*h))/h**2
            if o == 2:
                app[:] = (2*f(pts) - 5*f(pts-h) + 4*f(pts-2*h) - f(pts-3*h))/h**2
            if o == 3:
                app[:] = (35*f(pts)/12 - 26*f(pts-h)/3 + 19*f(pts-2*h)/2 - 
                          14*f(pts-3*h)/3 + 11*f(pts-4*h)/12)/h**2
    return app
