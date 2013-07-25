import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import matplotlib.pyplot as plt
import matplotlib.widgets as wg
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def exp_plot():
	x = np.linspace(-2, 3, 501)
	y = np.exp(x)
	plt.plot(x, y)
	plt.savefig("expplot.pdf")

def statemachine():
	x = np.linspace(1, 10, 10)
	y = np.random.rand(10, 10)

	plt.cla()
	for n in y:
		plt.plot(x, n)
	plt.savefig("statemachine.pdf")

def subplots():
	x = np.linspace(-np.pi, np.pi, 400)
	y1 = np.sin(x)
	y2 = np.cos(x)

	plt.subplot(211)
	plt.plot(x, y1)
	plt.subplot(212)
	plt.plot(x, y2)
	plt.savefig("subplots.pdf")

def three_d_plot():
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	x = np.linspace(-6, 6, 301)
	y = x.copy()
	X, Y = np.meshgrid(x, y)
	Z = np.sin(X)*np.sin(Y)
	ax.plot_surface(X, Y, Z)
	plt.savefig("3dplot.pdf")

def interact():
	def update(val):
		amp = samp.val
		freq = sfreq.val
		l.set_ydata(amp.np.sin(2*np.pi*freq*t))
		plt.draw()

	ax = plt.subplot(111)
	plt.subplots_adjust(bottom=.25)

	t = np.arange(0, 1, .001)
	a0, f0 = 5, 3
	s = a0*np.sin(2*np.pi*f0*t)
	l = plt.plot(t, s)

	plt.axis([0, 1, -10, 10])
	axfreq = plt.axes([.25, .1, .65, .03])
	axamp = plt.axes([.25, .15, .65, .03])
	sfreq = wg.Slider(axfreq, 'Freq', 0.1, 30.0, valinit=f0)
	samp = wg.Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)

	sfreq.on_changed(update)
	samp.on_changed(update)
	plt.savefig("interact.pdf")

if __name__ == "__main__":
	exp_plot()
	statemachine()
	subplots()
	interact()
	three_d_plot()
