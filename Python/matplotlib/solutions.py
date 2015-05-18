import numpy as np
from matplotlib import pyplot as plt
from matplotlib import widgets as wg
from mpl_toolkits.mplot3d import Axes3D
from mayavi import mlab

import os
import zipfile
import urllib

def problem1():
    X = np.linspace(0, 2*np.pi, 400)
    Y1 = np.sin(X)
    Y2 = np.cos(X)
    plt.plot(X, Y1, "r--", X, Y2, "b:")
    plt.show()

def problem2():
    X1 = np.linspace(-2, 1-.01, 300)
    X2 = np.linspace(1.01, 6, 500)
    f = lambda X: 1./(X - 1)
    Y1 = f(X1)
    Y2 = f(X2)
    plt.plot(X1, Y1, 'm--', X2, Y2, 'm--', linewidth=5.0)
    plt.ylim((-6,6))
    plt.show()

def problem3():
    X = np.linspace(0, 10, 1001)
    Y = np.sin(X) / (X + 1)
    plt.plot(X, Y, ":")
    plt.fill_between(X, Y, where=Y>0, color='b')
    plt.fill_between(X, Y, where=Y<=0, color='r')
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.title("My Plot")
    plt.grid()
    indices = np.zeros_like(X, dtype=bool)
    for i in xrange(1, Y.size-1):
        if (Y[i-1]-Y[i])*(Y[i]-Y[i+1])<0:
            indices[i]=True
    plt.scatter(X[indices], .5*Y[indices], marker='^')
    plt.xlim((0,10))
    plt.show()

def problem4():
    R = np.linspace(0, 2, 401)
    I = R.copy()
    R, I = np.meshgrid(R, I)
    X = R + complex(0,1)*I
    f = np.poly1d([1, 2, -1, 3])
    Y = np.absolute(f(X))
    plt.pcolormesh(R, I, Y)
    plt.show()
    
def heatmap():
    x = y = np.linspace(-2*np.pi, 2*np.pi, 402)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X)*np.sin(Y)/(X*Y)
    plt.pcolormesh(X, Y, Z, cmap='seismic')
    plt.pcolorbar()
    plt.gca().set_aspect('equal')
    plt.xlim([-2*np.pi, 2*np.pi])
    plt.ylim([-2*np.pi, 2*np.pi])
    plt.show()

def problem5():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = np.linspace(-10, 10, 501)
    y = np.linspace(-10, 10, 501)
    x, y = np.meshgrid(x, y)
    d = np.sqrt(x**2 + y**2)
    z = np.cos(d) / (.1 * d**2 + 1)
    ax.plot_surface(x, y, z)
    plt.show()

def problem6():
    ax = plt.subplot(111)
    plt.subplots_adjust(bottom=.25)
    t = np.arange(0, 1.01, .01)
    a = 5.
    f = 3.
    p = 0.
    fc = lambda a, f, p, t: a * np.sin(2*np.pi*f*(t-p))
    l = plt.plot(t, fc(a, f, p, t))[0]
    axfreq = plt.axes([.25, .05, .65, .03])
    axamp = plt.axes([.25, .1, .65, .03])
    axph = plt.axes([.25, .15, .65, .03])
    sfreq = wg.Slider(axfreq, "Freq", .1, 30., valinit=f)
    samp = wg.Slider(axamp, "Amp", .1, 10., valinit=a)
    sph = wg.Slider(axph, "Phase", 0, 2*np.pi, valinit=p)
    def update(val):
        a = samp.val
        f = sfreq.val
        p = sph.val
        l.set_ydata(fc(a,f,p,t))
        plt.draw()
    sfreq.on_changed(update)
    samp.on_changed(update)
    sph.on_changed(update)
    plt.show()

def problem7():
    x = np.linspace(-3,3)
    plt.subplot(221)
    plt.plot(x, np.exp(x))
    plt.subplot(222)
    plt.plot(x, np.sin(x))
    plt.subplot(223)
    plt.plot(x, np.cos(x))
    plt.subplot(224)
    plt.plot(x, x**2)
    plt.suptitle("My Different Plots")
    plt.show()
    
def subplot():
    x = np.random.rand(50)
    plt.subplot(1, 2, 1)
    plt.hist(x, bins=5, range=[0,1])
    
    plt.subplot(1,2,2)
    t = np.linspace(1,50,50)
    plt.scatter(t, x)
    
    tnew = np.linspace(0,51,2)
    y = x.mean()*np.ones(2)
    plt.plot(tnew, y, 'r', )
    
    plt.xlim([0,51])
    plt.show()

def problem8():
    opener = urllib.URLopener()
    opener.retrieve('https://s3.amazonaws.com/storage.enthought.com/www/sample_data/N36W113.hgt.zip', 'N36W113.hgt.zip')
    data = np.fromstring(zipfile.ZipFile('N36W113.hgt.zip').read('N36W113.hgt'), '>i2').reshape((3601, 3601)).astype('float32')
    data = data[:1000, 900:1900]
    data[data == -32768] = data[data>0].min()
    mlab.figure(size=(400, 320), bgcolor = (.16, .28, .46))
    mlab.surf(data, colormap="gist_earth", warp_scale=.2, vmin=1200, vmax=1610)
    mlab.view(-5.9, 83, 570, [5.3, 20, 238])
    return mlab.gcf()
