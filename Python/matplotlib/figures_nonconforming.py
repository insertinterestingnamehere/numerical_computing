import matplotlib.pyplot as plt
import numpy as np

# The plots "pcolor.png" and "pcolor2.png" weren't
# rendering nicely in the usual way.
# Run this script and save them from the interactive
# windows as png's if you ever want to regenerate them.

def pcolor_plot():
	n = 401
	x = np.linspace(-6, 6, n)
	y = np.linspace(-6, 6, n)
	X, Y = np.meshgrid(x, y)
	C = np.sin(X) * np.sin(Y)
	plt.pcolor(X, Y, C, )
	plt.show()

def pcolormesh_plot():
	R = np.linspace(0, 2, 401)
	I = R.copy()
	R, I = np.meshgrid(R, I)
	X = R + complex(0,1)*I
	f = np.poly1d([1, 2, -1, 3])
	Y = np.absolute(f(X))
	plt.pcolormesh(R, I, Y)
	plt.show()

if __name__ == "__main__":
	pcolor_plot()
	pcolormesh_plot()
