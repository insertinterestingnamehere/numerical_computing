import numpy as np
import matplotlib.pyplot as plt
from colorsys import hls_to_rgb

def colorize(z):
    zy=np.flipud(z)
    r = np.abs(zy)
    arg = np.angle(zy)

    h = (arg + np.pi)  / (2 * np.pi) + 0.5
    l = 1.0 - 1.0/(1.0 + r**.75)
    s = .8
    
    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0,2)
    c = c.swapaxes(0,1)
    return c
    
def plot_complex(f, xbounds, ybounds, res):
    x = np.linspace(xbounds[0], xbounds[1], res)
    y = np.linspace(ybounds[0], ybounds[1], res)
    X,Y = np.meshgrid(x,y)
    Z=f(X+Y*1j)
    plt.pcolormesh(X, Y, np.angle(Z), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    plt.show()
    
def plot_complex2(f, xbounds, ybounds, res):
    x = np.linspace(xbounds[0],xbounds[1], res)
    y = np.linspace(ybounds[0],ybounds[1], res)
    X,Y = np.meshgrid(x,y)
    Z=f(X+Y*1j)
    Zc = colorize(Z)
    plt.imshow(Zc, extent=(xbounds[0], xbounds[1], ybounds[0], ybounds[1]))
    plt.show()
    
################################################
################################################
# PLOTS

def check_plot():
    f = lambda x:np.sqrt(x**2+1)
    x = np.linspace(-3, 3, 401)
    y = np.linspace(-3, 3, 401)
    X,Y = np.meshgrid(x,y)
    Z=f(X+Y*1j)
    plt.pcolormesh(X, Y, np.angle(Z), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('check_plot.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    
def plot_id():
    f = lambda x:x
    x = np.linspace(-1, 1, 401)
    y = np.linspace(-1, 1, 401)
    X,Y = np.meshgrid(x,y)
    Z=f(X+Y*1j)
    plt.pcolormesh(X, Y, np.angle(Z), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('Identity.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    
def zeros():
    f = lambda x:x**3-1j*x**4-3*x**6
    x = np.linspace(-1, 1, 401)
    y = np.linspace(-1, 1, 401)
    X,Y = np.meshgrid(x,y)
    Z=f(X+Y*1j)
    plt.pcolormesh(X, Y, np.angle(Z), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('zeros.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    
def essential_pole():
    f = lambda x : np.exp(1/x)
    x = np.linspace(-1, 1, 803)
    y = np.linspace(-1, 1, 803)
    X,Y = np.meshgrid(x,y)
    Z=f(X+Y*1j)
    plt.pcolormesh(X, Y, np.angle(Z), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('essential_pole.png', bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__=='__main__':
	check_plot()
	plot_id()
	zeros()
	essential_pole()
