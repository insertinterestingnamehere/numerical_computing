from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import scipy as sp
import numpy as np
from solutions import get_vals

cmap = plt.cm.coolwarm
cmap_r = plt.cm.coolwarm_r

def sqrt_riemann_surface_1():
    """riemann surface for real part of sqrt(z)"""

    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = sp.meshgrid(X, Y)
    Z = sp.real(sp.sqrt(X+1j*Y))
    ax.plot_surface(X, Y, Z, cstride=1, rstride=1, linewidth=0, cmap=cmap)
    ax.plot_surface(X, Y, -Z, cstride=1, rstride=1, linewidth=0, cmap=cmap)
    plt.savefig('sqrt_riemann_surface_1.pdf', bbox_inches='tight', pad_inches=0)

def sqrt_riemann_surface_2():
    """riemann surface for imaginary part of sqrt(z)"""

    fig = plt.figure()
    ax = Axes3D(fig)
    X = sp.arange(-5, 5, 0.25)
    Y = sp.arange(-5, 0, 0.25)
    X, Y = sp.meshgrid(X, Y)
    Z = sp.imag(sp.sqrt(X+1j*Y))
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, cmap=cmap_r)
    ax.plot_surface(X, Y, -Z, rstride=1, cstride=1, linewidth=0, cmap=cmap)
    X = sp.arange(-5, 5, 0.25)
    Y = sp.arange(0,5,.25)
    X, Y = sp.meshgrid(X, Y)
    Z = sp.imag(sp.sqrt(X+1j*Y))
    ax.plot_surface(X, Y, -Z, rstride=1, cstride=1, linewidth=0, cmap=cmap_r)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, cmap=cmap)
    plt.savefig('sqrt_riemann_surface_2.pdf', bbox_inches='tight', pad_inches=0)
    
def log_riemann_surface():
    """riemann surface for imaginary part of ln(z)"""
    
    fig = plt.figure()
    ax = Axes3D(fig)
    X = sp.arange(-5, 5, 0.25)
    Y = sp.arange(-5, 0, 0.25)
    X, Y = sp.meshgrid(X, Y)
    Z = sp.imag(sp.log(X+1j*Y))
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, cmap=cmap_r)
    ax.plot_surface(X,Y,Z+2*sp.pi, rstride=1, cstride=1,linewidth=0, cmap=cmap_r)
    ax.plot_surface(X,Y,Z-2*sp.pi, rstride=1, cstride=1,linewidth=0, cmap=cmap_r)
    X = sp.arange(-5, 5, 0.25)
    Y = sp.arange(0,5,.25)
    X, Y = sp.meshgrid(X, Y)
    Z = sp.imag(sp.log(X+1j*Y))
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, cmap=cmap)
    ax.plot_surface(X,Y,Z+2*sp.pi, rstride=1, cstride=1,linewidth=0, cmap=cmap)
    ax.plot_surface(X,Y,Z-2*sp.pi, rstride=1, cstride=1,linewidth=0, cmap=cmap)
    plt.savefig('log_riemann_surface.pdf', bbox_inches='tight', pad_inches=0)
    
def arctan_riemann_surface():
    """Riemann surface for real part of arctan(z)"""
    
    fig = plt.figure()
    ax = Axes3D(fig)
    Xres, Yres = .01, .2
    ax.view_init(elev=11., azim=-56)
    X = sp.arange(-4, -.0001, Xres)
    Y = sp.arange(-4, 4, Yres)
    X, Y = sp.meshgrid(X, Y)
    Z = sp.real(sp.arctan(X+1j*Y))
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, cmap=cmap)
    ax.plot_surface(X, Y, Z+sp.pi, rstride=1, cstride=1, linewidth=0, cmap=cmap)
    ax.plot_surface(X, Y, Z-sp.pi, rstride=1, cstride=1, linewidth=0, cmap=cmap)
    X = sp.arange(.0001, 4, Xres)
    Y = sp.arange(-4,4, Yres)
    X, Y = sp.meshgrid(X, Y)
    Z = sp.real(sp.arctan(X+1j*Y))
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, cmap=cmap)
    ax.plot_surface(X, Y, Z+sp.pi, rstride=1, cstride=1, linewidth=0,cmap=cmap)
    ax.plot_surface(X, Y, Z-sp.pi, rstride=1, cstride=1, linewidth=0, cmap=cmap)
    plt.savefig('arctan_riemann_surface.pdf', bbox_inches='tight', pad_inches=0)

def poly_color_plot_real(p, res=101):
    X, Y, vals = get_vals(p, (-1, 1), (-1, 1), res=res)
    plt.pcolormesh(X, Y, vals.real)
    plt.savefig("poly_color_plot_real.pdf")

def poly_color_plot_imag(p, res=101):
    X, Y, vals = get_vals(p, (-1, 1), (-1, 1), res=res)
    plt.pcolormesh(X, Y, vals.imag)
    plt.savefig("poly_color_plot_imag.pdf")

def poly_surface_plot_real(p, res=101):
    X, Y, vals = get_vals(p, (-1, 1), (-1, 1), res)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, vals.real)
    plt.savefig("poly_surface_plot_real.pdf")

def poly_surface_plot_imag(p, res=101):
    X, Y, vals = get_vals(p, (-1, 1), (-1, 1), res)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, vals.imag)
    plt.savefig("poly_surface_plot_imag.pdf")

if __name__=='__main__':
    sqrt_riemann_surface_1()
    sqrt_riemann_surface_2()
    log_riemann_surface()
    arctan_riemann_surface()
    p = np.poly1d([1, 0, -1])
    poly_color_plot_real(p)
    poly_color_plot_imag(p)
    poly_surface_plot_real(p)
    poly_surface_plot_imag(p)
