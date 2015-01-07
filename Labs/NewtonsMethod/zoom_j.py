import scipy as sp
import matplotlib.pyplot as plt

# A class that will regenerate a fractal set as we zoom in, so that you
# can actually see the increasing detail.
class JuliaDisplay(object):
    def __init__(self, h=400, w=400, niter=20, power=2.0, c=0.4+0.124j):
        self.height = h
        self.width = w
        self.niter = niter
        self.power = power
        self.c = c

    def __call__(self, xstart, xend, ystart, yend):
        self.x = sp.linspace(xstart, xend, self.width)
        self.y = sp.linspace(ystart, yend, self.height)
        X,Y = sp.meshgrid(self.x,self.y)
        #c = 0.4+0.125j
        z = X+1.0j*Y
        for i in xrange(self.niter):
            z = z**self.power + self.c
            W = sp.exp(-abs(z))
        return W

    def ax_update(self, ax):
        ax.set_autoscale_on(False) # Otherwise, infinite loop

        #Get the number of points from the number of pixels in the window
        dims = ax.axesPatch.get_window_extent().bounds
        self.width = int(dims[2] + 0.5)
        self.height = int(dims[2] + 0.5)

        #Get the range for the new area
        xstart,ystart,xdelta,ydelta = ax.viewLim.bounds
        xend = xstart + xdelta
        yend = ystart + ydelta

        # Update the image object with our new data and extent
        im = ax.images[-1]
        im.set_data(self.__call__(xstart, xend, ystart, yend))
        im.set_extent((xstart, xend, ystart, yend))
        ax.figure.canvas.draw_idle()

jd = JuliaDisplay()
Z = jd(-1.5, 1.5, -1.5, 1.5)

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.imshow(Z, origin='lower', extent=(jd.x.min(), jd.x.max(), jd.y.min(), jd.y.max()))

ax1.callbacks.connect('xlim_changed', jd.ax_update)
ax1.callbacks.connect('ylim_changed', jd.ax_update)

plt.show()
