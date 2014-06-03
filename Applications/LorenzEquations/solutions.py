import numpy as np
from mayavi import mlab
from matplotlib import pyplot as plt
from numpy.random import rand, seed
from scipy.integrate import odeint
from scipy.stats import linregress

# ODE used in other functions.
def lorenz_ode((x, y, z), t, sigma=10., beta=8./3, rho=28.0):
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

def lorenz_plot(N=10, res=2000, t=10, seed_=120, atol=1E-15, rtol=1E-13,
                sigma=10., beta=8./3, rho=28.):
    """ Plot the trajectories given by the Lorenz equations for 'N' starting points.
    Choose random x, y, and z values between -15 and 15.
    Seed the random number generator with 'seed_'.
    Use a resolution of 'res' for the points in the plot.
    Plot the time values between 0 ant 't'.
    When computing the trajectories, pass the tolerance 'atol' to the ODE solver.
    Use different colors for each trajectory.
    Use the values of 'sigma', 'beta', and 'rho' in the Lorenz ODE. """
    
    # Get initial conditions.
    seed(seed_)
    x0 = -15 + 30 * rand(N, 3)
    
    # Solve for the trajectories.
    t = np.linspace(0, t, res)
    pts = np.empty((N, res, 3))
    for i, x in enumerate(x0):
        pts[i] = odeint(lorenz_ode, x, t,
                        args=(sigma, beta, rho), atol=atol, rtol=rtol)
    
    # Select the colors for the different curves.
    colors = np.zeros((N, 3))
    colors[:,1] = np.linspace(0, 1, N)
    colors = map(tuple, colors.tolist())
    
    # Plot the different trajectories.
    for x, color in zip(pts, colors):
        mlab.plot3d(x[:,0], x[:,1], x[:,2], tube_radius=.2, color=color)
    
    # Position the view for the plot.
    mlab.gcf().scene.camera.position = [127.23761585, -108.28736806, 6.35191272]
    mlab.gcf().scene.camera.focal_point = [-1.7792501449584961, -3.6287221908569336, 23.397351264953613]
    mlab.gcf().scene.camera.view_up = [-0.078467260964232038, -0.20339450183237351, 0.97594752194015633]
    mlab.gcf().scene.camera.clipping_range = [128.64624663718814, 328.22549479639167]
    
    # Show the plot.
    mlab.show()

def lorenz_animation(N=10, res=1000, step=2, t=10, seed_=120, atol=1E-15,
                     rtol=1E-13, delay=10, sigma=10., beta=8./3, rho=28.):
    """ Animate the trajectories given by the Lorenz equations for 'N' starting points.
    Choose random x, y, and z values between -15 and 15.
    Seed the random number generator with 'seed_'.
    Use a resolution of 'res' for the points in the plot.
    Plot the time values between 0 ant 't'.
    When computing the trajectories, pass the tolerances
    'atol' and 'rtol' to the ODE solver.
    At each update, add 'step' points to the plot.
    Use a delay of 'delay' at each update in the animation.
    Use different colors for each trajectory.
    Use the values of 'sigma', 'beta', and 'rho' in the Lorenz ODE. """
    
    # Get initial conditions.
    seed(seed_)
    x0 = -15 + 30 * rand(N, 3)
    
    # Solve for the trajectories.
    t = np.linspace(0, t, res)
    pts = np.empty((N, res, 3))
    for i, x in enumerate(x0):
        pts[i] = odeint(lorenz_ode, x, t,
                        args=(sigma, beta, rho), rtol=rtol, atol=atol)
    
    # Select the colors for the different curves.
    colors = np.zeros((N, 3))
    colors[:,1] = np.linspace(0, 1, N)
    colors = map(tuple, colors.tolist())
    
    # Plot the different trajectories.
    contours = [mlab.plot3d(x[:1,0], x[:1,1], x[:1,2], tube_radius=.15, color=color)
                for x, color in zip(pts, colors)]
    
    # Position the view for the plot.
    mlab.gcf().scene.camera.position = [127.23761585, -108.28736806, 6.35191272]
    mlab.gcf().scene.camera.focal_point = [-1.7792501449584961, -3.6287221908569336, 23.397351264953613]
    mlab.gcf().scene.camera.view_up = [-0.078467260964232038, -0.20339450183237351, 0.97594752194015633]
    mlab.gcf().scene.camera.clipping_range = [128.64624663718814, 328.22549479639167]
    
    # Define the animation.
    @mlab.show
    @mlab.animate(delay=delay)
    def trace_curve():
        for i in xrange(step, res, step):
            for c, x, color in zip(contours, pts, colors):
                c.mlab_source.reset(x=x[:i,0], y=x[:i,1], z=x[:i,2])
            yield
    
    # Run the animation.
    trace_curve()

def lorenz_tolerance_change(res=10000, step=5, t=50, seed_=120, atol1=1E-14,
                            atol2=1E-15, rtol1=1E-12, rtol2=1E-13, delay=10,
                            sigma=10., beta=8./3, rho=28.):
    """ Animate the trajectories given by the Lorenz equations.
    Plot two trajectories, one computed using the tolerances 'atol1' and 'rtol1',
    and one computed using the tolerances 'atol2' and 'rtol2'.
    Choose random x, y, and z values between -15 and 15.
    Seed the random number generator with 'seed_'.
    Use a resolution of 'res' for the points in the plot.
    Plot the time values between 0 ant 't'.
    At each update, add 'step' points to the plot.
    Use a delay of 'delay' at each update in the animation.
    Use different colors for each trajectory.
    Use the values of 'sigma', 'beta', and 'rho' in the Lorenz ODE. """
    # Get initial conditions.
    seed(seed_)
    x = -15 + 30 * rand(3)
    
    # Solve for the trajectories.
    # Plot them.
    t = np.linspace(0, t, res)
    y1 = odeint(lorenz_ode, x, t, args=(sigma, beta, rho), rtol=rtol1, atol=atol1)
    c1 = mlab.plot3d(y1[:1,0], y1[:1,1], y1[:1,2], tube_radius=.2, color=(1, 0, 0))
    y2 = odeint(lorenz_ode, x, t, args=(sigma, beta, rho), rtol=rtol2, atol=atol2)
    c2 = mlab.plot3d(y2[:1,0], y2[:1,1], y2[:1,2], tube_radius=.2, color=(0, 0, 1))
    
    # Position the view for the plot.
    mlab.gcf().scene.camera.position = [127.23761585, -108.28736806, 6.35191272]
    mlab.gcf().scene.camera.focal_point = [-1.7792501449584961, -3.6287221908569336, 23.397351264953613]
    mlab.gcf().scene.camera.view_up = [-0.078467260964232038, -0.20339450183237351, 0.97594752194015633]
    mlab.gcf().scene.camera.clipping_range = [128.64624663718814, 328.22549479639167]
    
    # Define the animation.
    @mlab.show
    @mlab.animate(delay=delay)
    def trace_curve():
        for i in xrange(step, res, step):
            c1.mlab_source.reset(x=y1[:i,0], y=y1[:i,1], z=y1[:i,2])
            c2.mlab_source.reset(x=y2[:i,0], y=y2[:i,1], z=y2[:i,2])
            yield
    
    # Run the animation.
    trace_curve()

def lorenz_perturbed(N=10, res=10000, step=5, t=50, seed_=120, atol=1E-15,
                     rtol=1E-13, epsilon=2.2e-16, delay=10,
                     sigma=10., beta=8./3, rho=28.):
    """ Animate the trajectories given by the Lorenz equations.
    Plot two trajectories, one with the initial value given by the
    random number generator after you seed it,
    and another that is equal to (1 + epsilon) times the other initial value.
    Choose random x, y, and z values between -15 and 15.
    Seed the random number generator with 'seed_'.
    Use a resolution of 'res' for the points in the plot.
    Plot the time values between 0 ant 't'.
    Pass the tolerances 'atol' and 'rtol' to the ODE solver.
    At each update, add 'step' points to the plot.
    Use a delay of 'delay' at each update in the animation.
    Use different colors for each trajectory.
    Use the values of 'sigma', 'beta', and 'rho' in the Lorenz ODE. """
    # Get initial conditions.
    seed(seed_)
    x1 = -15 + 30 * rand(3)
    x2 = x1 * (1. + epsilon)
    
    # Solve for the trajectories.
    # Plot them.
    t = np.linspace(0, t, res)
    y1 = odeint(lorenz_ode, x1, t, args=(sigma, beta, rho), atol=atol, rtol=rtol)
    c1 = mlab.plot3d(y1[:1,0], y1[:1,1], y1[:1,2], tube_radius=.2, color=(1, 0, 0))
    y2 = odeint(lorenz_ode, x2, t, args=(sigma, beta, rho), atol=atol, rtol=rtol)
    c2 = mlab.plot3d(y2[:1,0], y2[:1,1], y2[:1,2], tube_radius=.2, color=(0, 0, 1))
    
    # Position the view for the plot.
    mlab.gcf().scene.camera.position = [127.23761585, -108.28736806, 6.35191272]
    mlab.gcf().scene.camera.focal_point = [-1.7792501449584961, -3.6287221908569336, 23.397351264953613]
    mlab.gcf().scene.camera.view_up = [-0.078467260964232038, -0.20339450183237351, 0.97594752194015633]
    mlab.gcf().scene.camera.clipping_range = [128.64624663718814, 328.22549479639167]
    
    # Define the animation.
    @mlab.show
    @mlab.animate(delay=delay)
    def trace_curve():
        for i in xrange(2, res, step):
            c1.mlab_source.reset(x=y1[:i,0], y=y1[:i,1], z=y1[:i,2])
            c2.mlab_source.reset(x=y2[:i,0], y=y2[:i,1], z=y2[:i,2])
            yield
    
    # Run the animation.
    trace_curve()

def lyapunov_plot(res=10001, initial_time=10., t=10, seed_=5,
                  epsilon=1E-8, atol=1E-15, rtol=1E-13,
                  sigma=10., beta=8./3, rho=28.):
    """ Plot the separation between two trajectories through the Lorenz system.
    Use a logarithmic scale on the y-axis.
    Seed the random number generator with 'seed_'.
    Run the ODE solver through 'initial_time' using the given tolerances and resolution.
    Run the ODE solver an aditional 't' units of time on two new sets of initial conditions.
    One should be the final value of the previous computation.
    The other should be (1 + epsilon) times the other point.
    Use the resolutions 'res' and tolerances 'atol' and 'rtol' again
    when solving using the new initial values.
    Plot a fitting exponential curve through the points.
    On the log-scale, it will look like a line.
    Show the plot, and return the resulting approximation to the Lyapunov exponent.
    Use the values of 'sigma', 'beta', and 'rho' in the Lorenz ODE. """
    # Get starting points.
    seed(seed_)
    x1 = -15 + 30 * rand(3)
    # Run till the point is already in the attractor.
    x1 = odeint(lorenz_ode, x1, np.linspace(0, initial_time, res),
                args=(sigma, beta, rho), atol=atol, rtol=rtol)[-1]
    # Change it slightly.
    x2 = x1 * (1. + epsilon)
    
    # Find the trajectories.
    t = np.linspace(0, t, res)
    y1 = odeint(lorenz_ode, x1, t, atol=atol, rtol=rtol, args=(sigma, beta, rho))
    y2 = odeint(lorenz_ode, x2, t, atol=atol, rtol=rtol, args=(sigma, beta, rho))
    # Plot the separation.
    plt.semilogy(t, np.sqrt(((y1 - y2)**2).sum(axis=1)))
    
    # Compute the regression.
    slope, intercept, r_value, p_value, std_err = linregress(t, np.log(np.sqrt(((y1 - y2)**2).sum(axis=1))))
    # Compute the approximation.
    yapprox = slope * t + intercept
    # Plot the line.
    plt.semilogy(t, np.exp(yapprox))
    
    # Label the axes.
    plt.xlabel('Time')
    plt.ylabel('Separation')
    # Show it.
    plt.show()
    return slope
