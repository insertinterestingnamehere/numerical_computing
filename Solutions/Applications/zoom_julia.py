import scipy as sp
import matplotlib.pyplot as plt

class JuliaDisplay(object):
    def __init__(self, h=400, w=400, niter=50):
        self.height = h
        self.width = w
        self.niter = niter

    def __call__(self, interval):
        self.x = sp.linspace(interval[0], interval[1], self.width)
        self.y = sp.linspace(interval[0], interval[1], self.height).reshape(-1,1)
        c = self.x+1.0j*self.y
        threshold_time = sp.zeros((self.height, self.width))
        z = np.zeros(threshold_time.shape, dtype='complex128')
        mask = np.ones(threshold_time.shape, dtype='bool')
        for i in xrange(self.niter):
            z[mask] = z[mask]**2+c[mask]
            mask = (sp.absolute(z)<2.0)
            threshold_time += mask
        return threshold_time

    def ax_update(self, ax):
        ax.set_autoscale_on(False)

        dims = ax.axesPatch.get_window_extent().bounds
        self.width=int(dims[2]+0.5)
        self.height=int(dims[2]+0.5)

        xstart,ystart,xd,yd = ax.viewLim.bounds
        xend=xstart+xd
        yend=ystart+yd

        im=ax.images[-1]
        im.set_data(self.__call__([xstart
