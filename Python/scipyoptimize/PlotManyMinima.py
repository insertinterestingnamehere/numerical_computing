import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
import numpy as np
from scipy import optimize as opt
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def plot():

	rpts = 1000
	tpts = 100
	R = np.linspace(0., 4.5, rpts)
	Theta = np.linspace(0., np.pi, tpts)

	X = np.zeros([rpts,tpts])
	Y = np.zeros([rpts,tpts])
	Z = np.zeros([rpts,tpts])

	for i in range(0,rpts):
	    for ii in range(0,tpts):
		#print R[i],Theta[ii]
		X[i,ii] = R[i]*np.cos(Theta[ii])+1
		Y[i,ii] = R[i]*np.sin(Theta[ii])
		Z[i,ii] = R[i]**2*(1.0+ np.sin(4*R[i])**2)

	fig = plt.figure(figsize=(10,5))
	ax = fig.add_subplot(111, projection='3d',aspect='auto')
	ax.plot_surface(X, Y, Z,cmap=cm.winter)

	ax.set_xticks(range(-4,7))
	ax.set_yticks(range(0,5))

	plt.axis('tight')
	plt.show()
	#plt.savefig("ManyMinima.pdf",format='pdf')
	return 1;
