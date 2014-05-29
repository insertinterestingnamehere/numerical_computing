# import matplotlib
# matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
from solution import EmbeddingAlg

def plot_continuation_curve(c_list,Initial_Guess,F,line_type,plot_label=None):
	C, X = EmbeddingAlg(c_list,Initial_Guess,F)
	if plot_label==None:
		plt.plot(C,X,line_type,linewidth=2.5)
	else:
		plt.plot(C,X,line_type,linewidth=2.5,label=plot_label)
	return


def plot_axes():
	# T = np.linspace(0,5,500)
	# plt.plot(T,np.zeros(T.shape),'-k')   # Graph x and c axes
	# plt.plot(-T,np.zeros(T.shape),'-k')
	# plt.plot(np.zeros(T.shape),T,'-k')
	# plt.plot(np.zeros(T.shape),-T,'-k')
	# plt.ylabel('$x$',fontsize=16)
	# plt.axis([-4,4,-4,4])
	pass
	return


def SaddleNode_bifurcation():
	def F(x,c): return c + x**2
	
	# plot_axes()
	plot_continuation_curve(np.linspace(-4,0,700),np.sqrt(4),F,'--k',"Unstable equilibria") 
	plot_continuation_curve(np.linspace(-4,0,700),-np.sqrt(4),F,'-k',"Stable equilibria") 
	plt.title("Saddle-node bifurcation\n Prototype equation: $x' = \lambda + x^2$")
	plt.xlabel('$\lambda$',fontsize=16)
	# plt.savefig('SaddleNBifurcation.pdf')
	plt.plot([0],[0.],'ok',linewidth=2.0)
	plt.axis([-4,4,-4,4])
	plt.text(-3, 1.8, r'$x_1(\lambda)$', fontsize=15)
	plt.text(-3, -2.2, r'$x_2(\lambda)$', fontsize=15)
	plt.legend(loc='best')
	plt.show()
	plt.clf()
	
	x = np.linspace(-2.5,2.5,200)
	plt.plot(x,x**2 -2.,'-k',linewidth=2.0)
	plt.xlabel('$x$',fontsize=16)
	plt.ylabel('$\dot{x}$',fontsize=16)
	plt.title(r'$\dot{x} = 2 + x^2 $')
	T = np.linspace(0,5,500)
	plt.plot(T,np.zeros(T.shape),'-k')   # Graph x and c axes
	plt.plot(-T,np.zeros(T.shape),'-k')
	plt.plot(np.zeros(T.shape),T,'-k')
	plt.plot(np.zeros(T.shape),-T,'-k')
	# ax = plt.axes()
	plt.arrow(-.5, 0, -0.5, 0, head_width=0.1, head_length=0.1, fc='k', ec='k',linewidth=2.5)
	plt.arrow(1.7, 0, .5, 0, head_width=0.1, head_length=0.1, fc='k', ec='k',linewidth=2.5)
	plt.arrow(-2.3, 0, .5, 0, head_width=0.1, head_length=0.1, fc='k', ec='k',linewidth=2.5)
	# plt.show()
	plt.axis([-2.5,2.5,-3,5])
	plt.show()
	
	
	return


def Transcritical_bifurcation():
	def F(x,lmbda): return lmbda*x + x**2.
	
	plot_continuation_curve(np.linspace(-5,0,500),5,F,'-k') 
	plot_continuation_curve(np.linspace(-5,0,500),0,F,'-k') 
	plot_continuation_curve(np.linspace(5,0,500),-5,F,'-k') 
	plot_continuation_curve(np.linspace(5,0,500),0,F,'-k') 
	# plot_axes()
	# plt.title("Transcritical bifurcation\n Prototype equation: $x' = \lambda x + x^2$")
	plt.xlabel('$\lambda$',fontsize=16)
	plt.axis([-4,4,-4,4])
	plt.show()
	plt.clf()
	return


def Hysteresis():
	def F(x,lmbda):	return lmbda + x - x**3.
	
	plot_continuation_curve(np.linspace(-5,2,1400),-2,F,'-k')
	plot_continuation_curve(np.linspace(5,-2,1400),2,F,'-k') 
	plot_continuation_curve(np.linspace(0,-2,700),0,F,'-k') 
	plot_continuation_curve(np.linspace(0,2,700),0,F,'-k') 
	x0, x1 = 1./np.sqrt(3), -1./np.sqrt(3)
	l0,l1 = x0**3 -x0,x1**3 -x1
	plt.plot([l0,l1],[x0,x1],'ok')
	# plot_axes()
	plt.axis([-3,3,-3,3])
	# plt.title("Hysteresis loop\n Prototype equation: $x' = \lambda + x - x^3$")
	plt.xlabel('$\lambda$',fontsize=16)
	plt.show()
	plt.clf()	
	return


def Pitchfork_bifurcation():
	def F(x,lmbda): return lmbda*x - x**3.
	
	plot_continuation_curve(np.linspace(-5,0,1200),0,F,'-k')
	plot_continuation_curve(np.linspace(5,0,1200),-5,F,'-k')
	plot_continuation_curve(np.linspace(5,0,1200),5,F,'-k')
	plot_continuation_curve(np.linspace(5,0,1200),0,F,'-k')
	plot_axes()
	
	# plt.title("Pitchfork bifurcation\n Prototype equation: $x' = \lambda x - x^3$")
	plt.xlabel('$\lambda$',fontsize=16)
	plt.show()
	plt.clf()
	return


def Another_bifurcation(gamma):
	def F(x, lmbda): return gamma + lmbda*x - x**3.
	
	plot_continuation_curve(np.linspace(-5,0,1200),0,F,'-k') 
	plot_continuation_curve(np.linspace(5,0,1200),-5,F,'-k')
	plot_continuation_curve(np.linspace(5,0,1200),5,F,'-k')
	plot_continuation_curve(np.linspace(5,0,1200),0,F,'-k')
	plot_axes()
	# plt.title("A saddle-node bifurcation\nEquation: $x' = 1 + \lambda x - x^3$")
	plt.xlabel('$\lambda$',fontsize=16)
	plt.show()
	plt.clf()
	return





Another_bifurcation(-1)
Another_bifurcation(-.2)
Another_bifurcation(0)
Another_bifurcation(.2)
Another_bifurcation(1)
# SaddleNode_bifurcation()
# Hysteresis()
# Pitchfork_bifurcation()
# Transcritical_bifurcation()




