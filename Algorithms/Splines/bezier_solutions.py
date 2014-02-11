import numpy as np
from matplotlib import pyplot as plt
from matplotlib import widgets as wg
from scipy.misc import comb

# Decasteljau's algorithm problem
def decasteljau(p,t):
    n = p.shape[0]
    m = p.shape[1]
    q1 = p
    q2 = np.zeros((n, m))
    for i in xrange(n-1, 0, -1):
        q2 = np.zeros((i, m))
        for k in xrange(i):
            q2[k] = (1-t) * q1[k] + t * q1[k+1]
        q1 = q2
    return q2[0]

# used in interactive plot problem.
def decasteljau_animated(p, t, res=401):
    n = p.shape[0]
    plt.plot(p[:,0], p[:,1])
    plt.scatter(p[:,0], p[:,1])
    q1 = p
    q2 = np.zeros((n, 2))
    for i in xrange(n-1, 0, -1):
        q2 = np.zeros((i, 2))
        for k in xrange(i):
            q2[k] = (1-t) * q1[k] + t * q1[k+1]
        plt.plot(q2[:,0], q2[:,1])
        plt.scatter(q2[:,0], q2[:,1])
        q1 = q2
    # This line double-plots a point, but it will probably
    # be less costly to replot the last point than it
    # is to check on each iteration.
    plt.scatter(q2[:,0], q2[:,1], color='r', linewidth=3)
    t = np.linspace(0, 1, res)
    X = np.array([decasteljau(p,i) for i in t])
    plt.plot(X[:,0], X[:,1])
    pass

# interactive plot problem.
def decasteljau_interactive(n):
    ax = plt.subplot(1, 1, 1)
    plt.subplots_adjust(bottom=0.25)
    plt.axis([-1, 1, -1, 1])
    axT = plt.axes([0.25, 0.1, 0.65, 0.03])
    sT = wg.Slider(axT, 't', 0, 1, valinit=0)
    pts = plt.ginput(n, timeout=240)
    plt.subplot(1, 1, 1)
    pts = np.array(pts)
    plt.cla()
    T0=0
    decasteljau_animated(pts, T0)
    plt.axis([-1, 1, -1, 1])
    plt.draw()
    def update(val):
        T = sT.val
        ax.cla()
        plt.subplot(1, 1, 1)
        decasteljau_animated(pts,T)
        plt.xlim((-1, 1))
        plt.ylim((-1,1))
        plt.draw()
    sT.on_changed(update)
    plt.show()

# Bernstein polynomial problem
def bernstein(i, n):
    return comb(n, i, exact=True) * (np.poly1d([-1,1]))**(n-i) * (np.poly1d([1,0]))**i

# Coordinate function problem.
def bernstein_pt_aprox(X):
    n = X.shape[0]
    xpoly = np.poly1d([0])
    ypoly = np.poly1d([0])
    for i in xrange(n):
        npoly = bernstein(i, n-1)
        xpoly += X[i,0] * npoly
        ypoly += X[i,1] * npoly
    return xpoly, ypoly

if __name__ == '__main__':
    decasteljau_interactive(5)
