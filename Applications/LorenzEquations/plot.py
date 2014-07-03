import numpy as np
from mayavi import mlab
from scipy.integrate import odeint
from scipy.stats import linregress
from numpy.random import rand, seed

# Use our preset configuration for Matplotlib.
import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
from matplotlib import pyplot as plt

def helix(name, resolution=401):
    z = np.linspace(0, 2, resolution)
    x = np.cos(4 * np.pi * z)
    y = np.sin(4 * np.pi * z)
    c = mlab.plot3d(x, y, z, line_width=.2, color=(1, 0, 0))
    # Save the plot.
    mlab.savefig(name)
    mlab.clf()

def harmonic1(name):
    x = np.linspace(0, np.pi)
    y = np.linspace(0, np.pi)
    x, y = np.meshgrid(x, y, copy=False)
    z = np.sin(x) * np.sin(y)
    # Trick it into setting the perspective as if the figure were centered at the origin.
    f = mlab.mesh(x, y, np.zeros_like(z), scalars=z)
    f.mlab_source.set(z=z)
    # Zoom in a bit.
    mlab.gcf().scene.camera.position = mlab.gcf().scene.camera.position / 1.2
    # Save the plot.
    mlab.savefig(name)
    mlab.clf()

def harmonic2(name):
    x = np.linspace(0, np.pi)
    y = np.linspace(0, np.pi)
    x, y = np.meshgrid(x, y, copy=False)
    z = np.sin(2 * x) * np.sin(2 * y)
    mlab.mesh(x, y, z)
    # Zoom in a bit.
    mlab.gcf().scene.camera.position = mlab.gcf().scene.camera.position / 1.3
    # Save the plot.
    mlab.savefig(name)
    mlab.clf()

def lorenz_ode((x, y, z), t, sigma=10., beta=8./3, rho=28.0):
        return sigma * (y - x), x * (rho - z) - y, x * y - beta * z

def lorenz_plot(name, N=10, res=2000, step=2, t=10, seed_=120):
    # Select initial conditions
    seed(seed_)
    x0 = -15 + 30 * np.random.rand(N, 3)
    
    # Solve for the trajectories
    t = np.linspace(0, t, res)
    pts = np.empty((N, res, 3))
    for i, x in enumerate(x0):
        pts[i] = odeint(lorenz_ode, x, t)
    
    # Select the colors for the different curves.
    colors = np.zeros((N, 3))
    colors[:,1] = np.linspace(0, 1, N)
    colors = map(tuple, colors.tolist())
    
    # Plot the different trajectories.
    for x, color in zip(pts, colors):
        mlab.plot3d(x[:,0], x[:,1], x[:,2], tube_radius=.2, color=color)
    
    # Position the camera.
    mlab.gcf().scene.camera.position = np.array([165.40890060328016, -140.77357847515529, 8.2574865327247622]) / 1.55
    mlab.gcf().scene.camera.focal_point = [-1.7792501449584961, -3.6287221908569336, 23.397351264953613]
    mlab.gcf().scene.camera.view_up = [-0.078467260964232038, -0.20339450183237351, 0.97594752194015633]
    mlab.gcf().scene.camera.clipping_range = [128.64624663718814, 328.22549479639167]
    
    # Save the plot.
    mlab.savefig(name)
    mlab.clf()

def lyapunov_plot(name, res=10001, initial_time=10., t=10, seed_=5,
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
    # Save the plot.
    plt.savefig(name)
    plt.clf()

if __name__ == '__main__':
    helix("helix.png")
    harmonic1('harmonic1.png')
    harmonic2('harmonic2.png')
    lorenz_plot('lorenz_plot.png')
    lyapunov_plot('lyapunov_plot.pdf')
    
