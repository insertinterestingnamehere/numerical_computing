import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import matplotlib.pyplot as plt
import matplotlib.widgets as wg
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from mayavi import mlab
import solutions

png_size = (1024, 768)

def exp_plot():
    x = np.linspace(-2, 3, 501)
    y = np.exp(x)
    plt.plot(x, y)
    plt.savefig("exp_plot.pdf", bbox_inches='tight')
    plt.clf()

def statemachine():
    x = np.linspace(1, 10, 10)
    y = np.random.rand(10, 10)

    plt.cla()
    for n in y:
        plt.plot(x, n)
    plt.savefig("statemachine.pdf", bbox_inches='tight')
    plt.clf()

def histogram():
    x = array([ 3,  9,  5,  2,  7, 10,  7,  2,  5,  8,  7,  8,  1,  9,  7, 10,  2,
        1,  9,  2])
    plt.hist(x, bins=10, range=[.5, 10.5])
    plt.xlim([.5, 10.5])
    plt.savefig('histogram.pdf', bbox_inches='tight')
    plt.close()
    
def scatter():
    x = array([ 3,  9,  5,  2,  7, 10,  7,  2,  5,  8,  7,  8,  1,  9,  7, 10,  2,
        1,  9,  2])
    t = np.linspace(1,20,20)
    plt.scatter(t, x, s=100)
    plt.xlim([0,21])
    plt.savefig('scatter.pdf', bbox_inches='tight')

def subplots():
    x = np.linspace(-np.pi, np.pi, 400)
    y1 = np.sin(x)
    y2 = np.cos(x)

    plt.subplot(211)
    plt.plot(x, y1)
    plt.xlim([-np.pi, np.pi])
    
    plt.subplot(212)
    plt.plot(x, y2)
    plt.xlim([-np.pi, np.pi])
    
    plt.savefig("subplots.pdf", bbox_inches='tight')
    plt.clf()
    
def sinxsiny():
    n = 401 
    x = np.linspace(-6, 6, n) 
    y = np.linspace(-6, 6, n) 
    X, Y = np.meshgrid(x, y) # returns a coordinate matrix given coordinate vectors. 
    C = np.sin(X) * np.sin(Y) 
    
    
    plt.pcolormesh(X, Y, C, edgecolors='face', shading='flat')
    plt.colorbar()
    plt.savefig("sinxsiny.png", size=png_size)
    plt.clf()
    
def pcolor2():
    R = np.linspace(0, 2, 401)
    I = R.copy()
    R, I = np.meshgrid(R, I)
    X = R + complex(0,1)*I
    f = np.poly1d([1, 2, -1, 3])
    Y = np.absolute(f(X))
    plt.pcolormesh(R, I, Y, edgecolors='face', shading='flat')
    plt.savefig('pcolor2.png', size=png_size)
    plt.clf()

def three_d_plot():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = np.linspace(-6, 6, 301)
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X)*np.sin(Y)
    ax.plot_surface(X, Y, Z)
    plt.savefig("3dplot.pdf")
    plt.clf()

def interact():

    ax = plt.subplot(111)
    plt.subplots_adjust(bottom=.25)

    t = np.arange(0, 1, .001)
    a0, f0 = 5, 3
    s = a0*np.sin(2*np.pi*f0*t)
    l = plt.plot(t, s)[0]

    plt.axis([0, 1, -10, 10])
    axfreq = plt.axes([.25, .05, .65, .03])
    axamp = plt.axes([.25, .1, .65, .03])
    sfreq = wg.Slider(axfreq, 'Freq', 0.1, 30.0, valinit=f0)
    samp = wg.Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)
    
    def update(val):
        amp = samp.val
        freq = sfreq.val
        l.set_ydata(amp*np.sin(2*np.pi*freq*t))
        plt.draw()

    sfreq.on_changed(update)
    samp.on_changed(update)
    plt.savefig("interact.pdf")
    plt.clf()

def plot3d():
    num = np.pi/1000
    pts = np.arange(0, 2*np.pi + num, num)
    x = np.cos(pts) * (1 + np.cos(pts*6))
    y = np.sin(pts) * (1 + np.cos(pts*6))
    z = np.sin(pts*6/11)
    mlab.plot3d(x, y, z)
    mlab.savefig("plot3d.png", size=png_size)
    mlab.clf()
    
def points3d():
    pts = np.linspace(0, 4 * np.pi, 30)
    x = np.sin(2 * pts)
    y = np.cos(pts)
    z = np.cos(2 * pts)
    s = 2+np.sin(pts)
    mlab.points3d(x, y, z, s, colormap="cool", scale_factor=.15)
    mlab.savefig("points3d.png", size=png_size)
    mlab.clf()
    
def GrandCanyon():
    f = solutions.problem8()
    mlab.savefig("GrandCanyon.png", size=png_size)
    mlab.clf()
    
def fancymesh():
    mlab.savefig('fancymesh.png', size=png_size, figure=mlab.test_fancy_mesh())
    mlab.clf()
    
def prob3_solution():
    f = solutions.problem3()
    plt.savefig('soln3.pdf')
    plt.clf()
    
def prob2_solution():
    f = solutions.problem2()
    plt.savefig('soln2.pdf', bbox_inches='tight')
    plt.clf()
    
def subplot_solution():
    f = solutions.subplot()
    plt.savefig('subplotProb.pdf', bbox_inches='tight')
    plt.clf()
    
def heatmap_solution():
    f = solutions.heatmap()
    plt.savefig('pcolor2.png', bbox_inches='tight')
    plt.clf()
    
if __name__ == "__main__":
    exp_plot()
    statemachine()
    subplots()
    scatter()
    histogram()
    interact()
    three_d_plot()
    sinxsiny()
    pcolor2()
    plot3d()
    points3d()
    fancymesh()
    GrandCanyon()
    prob3_solution()
    prob2_solution()
