import numpy as np

def initialize_all(y0, t0, t, n):
    """ An initialization routine for the different ODE solving
    methods in the lab. This initializes Y, T, and h. """
    if isinstance(y0, np.ndarray):
        Y = np.empty((n, y.size)).squeeze()
    else:
        Y = np.empty(n)
    Y[0] = y0
    T = np.linspace(t0, t, n)
    h = float(t - t0) / (n - 1)
    return Y, T, h

def euler(f, y0, t0, t, n):
    """ Use the Euler method to compute an approximate solution
    to the ODE y' = f(t, y) at n equispaced parameter values from t0 to t
    with initial conditions y(t0) = y0.
    
    y0 is assumed to be either a constant or a one-dimensional numpy array.
    t and t0 are assumed to be constants.
    f is assumed to accept two arguments.
    The first is a constant giving the value of t.
    The second is a one-dimensional numpy array of the same size as y.
    
    This function returns an array Y of shape (n,) if
    y is a constant or an array of size 1.
    It returns an array of shape (n, y.size) otherwise.
    In either case, Y[i] is the approximate value of y at
    the i'th value of np.linspace(t0, t, n).
    """
    Y, T, h = initialize_all(y0, t0, t, n)
    for i in xrange(1, n):
        Y[i] = Y[i-1] + f(T[i-1], Y[i-1]) * h
    return Y

def euler_accuracy():
    """ Test the accuracy of the Euler method using the
    initial value problem y' + y = 2 - 2x, 0 <= x <= 2, y(0) = 0.
    Plot your solutions over the given domain with n as 11, 21, and 41.
    Also plot the exact solution. 
    Show the plot. """
    f = lambda x, y: 2 - y - 2 * x
    for n in (11, 21, 41):
        T = np.linspace(0, 2, n)
        plt.plot(T, euler(f, 0, 0, 2, n))
    plt.plot(T, 4 - 2 * T - 4 * np.exp(-T))
    plt.show()

# The inversion here could also be done using scipy.optimize's Newton's method.
# Currently in the lab, this function isn't required.
def backwards_euler(f, fsolve, y0, t0, t, n):
    """ Use the backward Euler method to compute an approximate solution
    to the ODE y' = f(t, y) at n equispaced parameter values from t0 to t
    with initial conditions y(t0) = y0.
    
    y0 is assumed to be either a constant or a one-dimensional numpy array.
    t and t0 are assumed to be constants.
    f is assumed to accept two arguments.
    The first is a constant giving the value of t.
    The second is a one-dimensional numpy array of the same size as y.
    fsolve is a function that solves the equation
    y(x_{i+1}) = y(x_i) + h f(x_{i+1}, y(x_{i+1}))
    for the appropriate value for y(x_{i+1}).
    It should accept three arguments.
    The first should be the value of y(x_i).
    The second should be the distance between values of t.
    The third should be the value of x_{i+1}.
    
    This function returns an array Y of shape (n,) if
    y is a constant or an array of size 1.
    It returns an array of shape (n, y.size) otherwise.
    In either case, Y[i] is the approximate value of y at
    the i'th value of np.linspace(t0, t, n).
    """
    Y, T, h = initialize_all(y0, t0, t, n)
    for i in xrange(1, n):
        Y[i] = fsolve(Y[i-1], h, T[i])
    return Y

def midpoint(f, y0, t0, t, n):
    """ Use the midpoint method to compute an approximate solution
    to the ODE y' = f(t, y) at n equispaced parameter values from t0 to t
    with initial conditions y(t0) = y0.
    
    y0 is assumed to be either a constant or a one-dimensional numpy array.
    t and t0 are assumed to be constants.
    f is assumed to accept two arguments.
    The first is a constant giving the value of t.
    The second is a one-dimensional numpy array of the same size as y.
    
    This function returns an array Y of shape (n,) if
    y is a constant or an array of size 1.
    It returns an array of shape (n, y.size) otherwise.
    In either case, Y[i] is the approximate value of y at
    the i'th value of np.linspace(t0, t, n).
    """
    Y, T, h = initialize_all(y0, t0, t, n)
    for i in xrange(1, n):
        Y[i] = Y[i-1] + h * f(T[i-1] + h / 2., Y[i-1] + (h / 2.) * f(T[i-1], Y[i-1]))
    return Y

# This one isn't currently required in the lab.
def modified_euler(f, y0, t0, t, n):
    """ Use the modified Euler method to compute an approximate solution
    to the ODE y' = f(t, y) at n equispaced parameter values from t0 to t
    with initial conditions y(t0) = y0.
    
    y0 is assumed to be either a constant or a one-dimensional numpy array.
    t and t0 are assumed to be constants.
    f is assumed to accept two arguments.
    The first is a constant giving the value of t.
    The second is a one-dimensional numpy array of the same size as y.
    
    This function returns an array Y of shape (n,) if
    y is a constant or an array of size 1.
    It returns an array of shape (n, y.size) otherwise.
    In either case, Y[i] is the approximate value of y at
    the i'th value of np.linspace(t0, t, n).
    """
    Y, T, h = initialize_all(y0, t0, t, n)
    for i in xrange(1, n):
        Y[i] = Y[i-1] + (h / 2.) * (f(T[i-1], Y[i-1]) + f(T[i-1], Y[i-1] + h * f(T[i-1], Y[i-1])))
    return Y

def RK4(f, y0, t0, t, n):
    """ Use the RK4 method to compute an approximate solution
    to the ODE y' = f(t, y) at n equispaced parameter values from t0 to t
    with initial conditions y(t0) = y0.
    
    y0 is assumed to be either a constant or a one-dimensional numpy array.
    t and t0 are assumed to be constants.
    f is assumed to accept two arguments.
    The first is a constant giving the value of t.
    The second is a one-dimensional numpy array of the same size as y.
    
    This function returns an array Y of shape (n,) if
    y is a constant or an array of size 1.
    It returns an array of shape (n, y.size) otherwise.
    In either case, Y[i] is the approximate value of y at
    the i'th value of np.linspace(t0, t, n).
    """
    Y, T, h = initialize_all(y0, t0, t, n)
    for i in xrange(1, n):
        K1 = f(T[i-1], Y[i-1])
        tplus = (T[i] + T[i-1]) * .5
        K2 = f(tplus, Y[i-1] + .5 * h * K1)
        K3 = f(tplus, Y[i-1] + .5 * h * K2)
        K4 = f(T[i], Y[i-1] + h * K3)
        Y[i] = Y[i-1] + (h / 6.) * (K1 + 2 * K2 + 2 * K3 + K4)
    return Y

def compare_accuracies(N):
    """ Test the accuracies of the Euler, backwards Euler, modified Euler,
    midpoint, and RK4 methods using initial value problem
    y' + y = 2 - 2x, 0 <= x <= 2, y(0) = 0.
    Use the different values of n in N, and plot h=1./(n-1)
    vs the RELATIVE error at time t. """
    t = 2
    f = lambda x, y: 2 - y - 2 * x
    exact = 4 - 2 * t - 4 * np.exp(-t)
    euler_err = [abs((euler(f, 0, 0, t, n)[-1] - exact) / exact) for n in N]
    midpoint_err = [abs((midpoint(f, 0, 0, t, n)[-1] - exact) / exact) for n in N]
    RK4_err = [abs((RK4(f, 0, 0, t, n)[-1] - exact) / exact) for n in N]
    H = [1. / (n-1) for n in N]
    plt.loglog(H, euler_err, H, midpoint_err, H, RK4_err)
    plt.show()

def simple_harmonic_oscillator(y0, t0, t, n, m=1, k=1):
    """ Use the RK4 method to solve for the simple harmonic oscillator
    problem described in the problem about simple harmonic oscillators.
    Return the array of values at the equispaced points. """
    f = lambda x, y: np.array([y[1], - k * y[0] / float(m)])
    return RK4(f, y0, t0, t, n)[:,0]

def damped_harmonic_oscillator(y0, t0, t, n, gamma):
    """ Use the RK4 method to solve for the damped harmonic oscillator
    problem described in the problem about damped harmonic oscillators.
    Return the array of values at the equispaced points. """
    gamma = .5
    f = lambda x, y: np.array([y[1], - (gamma * y[1] + y[0])])
    return RK4(f, y0, t0, t, n)[:,0]

def forced_harmonic_oscillator(y0, t0, t, n, gamma, omega):
    """ Use the RK4 method to solve for the forced harmonic oscillator
    problem. y0, t0, t, and n are the same as the variables passed
    to the RK4 function. gamma and omega are constants. """
    f = lambda x, y: np.array([y[1], np.cos(omega * x) - y[0] - gamma * y[1] / 2])
    return RK4(f, y0, t0, t, n)[:,0]
