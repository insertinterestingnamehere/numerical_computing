import numpy as np
try:
    from mayavi import mlab as ml
    use_mayavi = True
except:
    use_mayavi = False
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

def get_vals(f, xbounds, ybounds, res):
    x = np.linspace(xbounds[0], xbounds[1], res)
    y = np.linspace(ybounds[0], ybounds[1], res)
    X, Y = np.meshgrid(x, y, copy=False)
    return X, Y, f(X + 1.0j * Y)

def plot_poly_real_mayavi(p, xbounds=(-1, 1), ybounds=(-1, 1), res=401):
    X, Y, vals = get_vals(p, xbounds, ybounds, res)
    ml.mesh(X, Y, vals.real)
    ml.show()

def plot_poly_imag_mayavi(p, xbounds=(-1, 1), ybounds=(-1, 1), res=401):
    X, Y, vals = get_vals(p, xbounds, ybounds, res)
    ml.mesh(X, Y, vals.imag)
    ml.show()

def plot_poly_both_mayavi(p, xbounds=(-1, 1), ybounds=(-1, 1), res=401):
    X, Y, vals = get_vals(p, xbounds, ybounds, res)
    ml.mesh(X, Y, vals.real)
    ml.mesh(X, Y, vals.imag)
    ml.show()

def plot_poly_real_matplotlib(p, xbounds=(-1, 1), ybounds=(-1, 1), res=401):
    X, Y, vals = get_vals(p, xbounds, ybounds, res)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, vals.real)
    plt.show()

def plot_poly_imag_matplotlib(p, xbounds=(-1, 1), ybounds=(-1, 1), res=401):
    X, Y, vals = get_vals(p, xbounds, ybounds, res)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, vals.imag)
    plt.show()

def plot_poly_both_matplotlib(p, xbounds=(-1, 1), ybounds=(-1, 1), res=401):
    X, Y, vals = get_vals(p, xbounds, ybounds, res)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, vals.real)
    ax.plot_surface(X, Y, vals.imag)
    plt.show()

def nroot_real_matplotlib(n, res=101):
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

def nroot_imag_matplotlib(n, res=101):
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

def nroot_real_mayavi(n, res=401):
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

def nroot_imag_mayavi(n, res=401):
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
    p = np.poly1d([1, 0, 0, 0, 1])
    if use_mayavi:
        nroot_real_mayavi(5)
        nroot_imag_mayavi(5)
        plot_poly_both_mayavi(p)
    else:
        nroot_real_matplotlib(5)
        nroot_imag_matplotlib(5)
        plot_poly_both_matplotlib(p)
