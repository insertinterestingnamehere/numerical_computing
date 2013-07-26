import numpy as np
import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
from matplotlib import pyplot as plt

t = np.linspace(0,1,401)
cx = np.empty(801)
cy = np.empty(801)
r = 2
cx[:401] =  2 * r * t - r
cy[:401] = 0.
cx[400:] = r * np.cos(np.pi * t)
cy[400:] = r * np.sin(np.pi * t)
plt.xlim((-3,3))
plt.ylim((-1,3))
plt.axes()
plt.plot(cx, cy)
plt.savefig("contour1.pdf")
plt.cla()

cx = np.empty(1601)
cy = np.empty(1601)
cx[:401] = (r - 1./r) * t + 1./r
cy[:401] = 0.
cx[800:1201] = -cx[:401]
cy[800:1201] = 0.
cx[400:801] = r * np.cos(np.pi * t)
cy[400:801] = r * np.sin(np.pi * t)
cx[1200:] = np.cos(np.pi * t[::-1]) / r
cy[1200:] = np.sin(np.pi * t[::-1]) / r
plt.xlim((-3,3))
plt.ylim((-1,3))
plt.axes()
plt.plot(cx, cy)
plt.savefig("contour2.pdf")
