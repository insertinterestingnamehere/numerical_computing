import numpy as np
import cmath as cm
from sympy import mpmath as mp

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

def contour_int(f, c, t0, t1):
    return complex(mp.quad(lambda t: f(c(t)) * mp.diff(c, t), (t0, t1)))

def cauchy_verify(f, c, z0, t0, t1):
    g = lambda z: f(z) / (z - z0)
    return contour_int(g, c, t0, t1)

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
    
    f1 = lambda z: z.conjugate()
    f2 = lambda z: mp.exp(z)
    c1 = lambda t: mp.cos(t) + 1.0j * mp.sin(t)
    c2 = lambda t: t + 1.0j * t
    c3 = lambda t: t
    c4 = lambda t: 1 + 1.0j * t
    c5 = lambda t: mp.cos(t) + 1.0j + 1.0j * mp.sin(t)
    print "z conjugate counterclockwise along the unit ball starting and ending at 1."
    print contour_int(f1, c1, 0, 2 * np.pi)
    print "z conjugate along a straight line from 0 to 1+1j."
    print contour_int(f1, c2, 0, 1)
    print "z conjugate along the real axis from 0 to 1, then along the line from 1 to 1+1j."
    print contour_int(f1, c3, 0, 1) + contour_int(f1, c4, 0, 1)
    print "z conjugate along the unit ball centered at 1j from 0 to 1+1j."
    print contour_int(f1, c5, 0, 2 * np.pi)
    print "e^z counterclockwise along the unit ball starting and ending at 1."
    print contour_int(f2, c1, 0, 2 * np.pi)
    print "e^z along a straight line from 0 to 1+1j."
    print contour_int(f2, c2, 0, 1)
    print "e^z along the real axis from 0 to 1, then along the line from 1 to 1+1j."
    print contour_int(f2, c3, 0, 1) + contour_int(f2, c4, 0, 1)
    print "e^z along the unit ball centered at 1j from 0 to 1+1j"
    print contour_int(f2, c5, 0, 2 * np.pi)
