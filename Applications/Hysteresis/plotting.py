# import matplotlib
# matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
from solution import EmbeddingAlg

def plot_continuation_curve(c_list,Initial_Guess,F):
	C, X = EmbeddingAlg(c_list,Initial_Guess,F)
	plt.plot(C,X,'-k',linewidth=2.5)
	return


def plot_axes():
	# T = np.linspace(0,5,500)
	# plt.plot(T,np.zeros(T.shape),'-k')   # Graph x and c axes
	# plt.plot(-T,np.zeros(T.shape),'-k')
	# plt.plot(np.zeros(T.shape),T,'-k')
	# plt.plot(np.zeros(T.shape),-T,'-k')
	# plt.ylabel('$x$',fontsize=16)
	plt.axis([-5,5,-5,5])
	return


def SaddleNode_bifurcation():
	def F(x,c): return c + x**2
	
	plot_axes()
	plot_continuation_curve(np.linspace(-5,0,500),np.sqrt(5),F) 
	plot_continuation_curve(np.linspace(-5,0,500),-np.sqrt(5),F) 
	plt.title("Saddle-node bifurcation\n Prototype equation: $x' = c + x^2$")
	plt.xlabel('$\lambda$',fontsize=16)
	# plt.savefig('SaddleNBifurcation.pdf')
	plt.show()
	plt.clf()
	return


def Transcritical_bifurcation():
	def F(x,lmbda): return lmbda*x + x**2.
	
	plot_continuation_curve(np.linspace(-5,0,500),5,F) 
	plot_continuation_curve(np.linspace(-5,0,500),0,F) 
	plot_continuation_curve(np.linspace(5,0,500),-5,F) 
	plot_continuation_curve(np.linspace(5,0,500),0,F) 
	plot_axes()
	plt.title("Transcritical bifurcation\n Prototype equation: $x' = \lambda x + x^2$")
	plt.xlabel('$\lambda$',fontsize=16)
	plt.show()
	plt.clf()
	return


def Hysteresis():
	def F(lmbda,x):	return lmbda + x - x**3.
	
	plot_continuation_curve(np.linspace(-5,2,1400),-2,F)
	plot_continuation_curve(np.linspace(5,-2,1400),2,F) 
	plot_continuation_curve(np.linspace(0,-2,700),0,F) 
	plot_continuation_curve(np.linspace(0,2,700),0,F) 
	plot_axes()
	plt.title("Hysteresis loop\n Prototype equation: $x' = \lambda + x - x^3$")
	plt.xlabel('$\lambda$',fontsize=16)
	plt.show()
	plt.clf()	
	return


def Pitchfork_bifurcation():
	def F(x,lmbda): return lmbda*x - x**3.
	
	plot_continuation_curve(np.linspace(-5,0,1200),0,F)
	plot_continuation_curve(np.linspace(5,0,1200),-5,F)
	plot_continuation_curve(np.linspace(5,0,1200),5,F)
	plot_continuation_curve(np.linspace(5,0,1200),0,F)
	plot_axes()
	
	plt.title("Pitchfork bifurcation\n Prototype equation: $x' = \lambda x - x^3$")
	plt.xlabel('$\lambda$',fontsize=16)
	plt.show()
	plt.clf()
	return


def Another_bifurcation(n):
	def F(x, lmbda): return n + lmbda*x - x**3.
	
	plot_continuation_curve(np.linspace(-5,0,1200),0,F) 
	plot_continuation_curve(np.linspace(5,0,1200),-5,F)
	plot_continuation_curve(np.linspace(5,0,1200),5,F)
	plot_continuation_curve(np.linspace(5,0,1200),0,F)
	plot_axes()
	plt.title("A saddle-node bifurcation\nEquation: $x' = 1 + \lambda x - x^3$")
	plt.xlabel('$\lambda$',fontsize=16)
	plt.show()
	plt.clf()
	return





Another_bifurcation(1)
# SaddleNode_bifurcation()
# Hysteresis()
# Pitchfork_bifurcation()
# Transcritical_bifurcation()




