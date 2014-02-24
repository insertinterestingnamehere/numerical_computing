import numpy as np
try:
    from mayavi import mlab as ml
    use_mayavi = True
except:
    use_mayavi = False
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

def plot_real_part_matplotlib(n, res=101):
    x = np.linspace(-1, 1, res)
    X, Y = np.meshgrid(x, x, copy=False)
    Z = X + 1.0j * Y
    r = np.absolute(Z)
    theta = np.angle(Z)
    rroot = r**(1./n)
    theta /= n
    real = rroot * np.cos(theta)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X[x>=0], Y[x>=0], real[x>=0])
    ax.plot_surface(X[x<0], Y[x<0], real[x<0])
    for i in xrange(1, n):
        theta += 2. * np.pi / n
        real = rroot * np.cos(theta)
        ax.plot_surface(X[x>=0], Y[x>=0], real[x>=0])
        ax.plot_surface(X[x<0], Y[x<0], real[x<0])
    plt.show()

def plot_imag_part_matplotlib(n, res=101):
    x = np.linspace(-1, 1, res)
    X, Y = np.meshgrid(x, x, copy=False)
    Z = X + 1.0j * Y
    r = np.absolute(Z)
    theta = np.angle(Z)
    rroot = r**(1./n)
    theta /= n
    imag = rroot * np.sin(theta)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X[x>=0], Y[x>=0], imag[x>=0])
    ax.plot_surface(X[x<0], Y[x<0], imag[x<0])
    for i in xrange(1, n):
        theta += 2. * np.pi / n
        imag = rroot * np.sin(theta)
        ax.plot_surface(X[x>=0], Y[x>=0], imag[x>=0])
        ax.plot_surface(X[x<0], Y[x<0], imag[x<0])
    plt.show()

def plot_real_part_mayavi(n, res=401):
    x = np.linspace(-1, 1, res)
    X, Y = np.meshgrid(x, x, copy=False)
    Z = X + 1.0j * Y
    r = np.absolute(Z)
    theta = np.angle(Z)
    rroot = r**(1./n)
    theta /= n
    real = rroot * np.cos(theta)
    ml.mesh(X[x>=0], Y[x>=0], real[x>=0], colormap='Blues')
    ml.mesh(X[x<0], Y[x<0], real[x<0], colormap='Blues')
    for i in xrange(1, n):
        theta += 2. * np.pi / n
        real = rroot * np.cos(theta)
        ml.mesh(X[x>=0], Y[x>=0], real[x>=0], colormap='Blues')
        ml.mesh(X[x<0], Y[x<0], real[x<0], colormap='Blues')
    ml.show()

def plot_imag_part_mayavi(n, res=401):
    x = np.linspace(-1, 1, res)
    X, Y = np.meshgrid(x, x, copy=False)
    Z = X + 1.0j * Y
    r = np.absolute(Z)
    theta = np.angle(Z)
    rroot = r**(1./n)
    theta /= n
    imag = rroot * np.sin(theta)
    ml.mesh(X[x>=0], Y[x>=0], imag[x>=0], colormap='Blues')
    ml.mesh(X[x<0], Y[x<0], imag[x<0], colormap='Blues')
    for i in xrange(1, n):
        theta += 2. * np.pi / n
        imag = rroot * np.sin(theta)
        ml.mesh(X[x>=0], Y[x>=0], imag[x>=0], colormap='Blues')
        ml.mesh(X[x<0], Y[x<0], imag[x<0], colormap='Blues')
    ml.show()

if __name__ == '__main__':
    if use_mayavi:
        plot_real_part_mayavi(5)
        plot_imag_part_mayavi(5)
    else:
        plot_real_part_matplotlib(5)
        plot_imag_part_matplotlib(5)
