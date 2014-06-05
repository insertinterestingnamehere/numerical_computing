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
		# pass
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
	plt.plot([0],[0.],'ok',linewidth=2.0,markersize=7.)
	plt.axis([-4,4,-4,4])
	plt.text(-3, 1.8, r'$x_1(\lambda)$', fontsize=15)
	plt.text(-3, -2.2, r'$x_2(\lambda)$', fontsize=15)
	plt.legend(loc='best')
	plt.savefig('SaddleNBifurcation.pdf')
	# plt.show()
	plt.clf()
	
	x = np.linspace(-2.5,2.5,200)
	plt.plot(x,x**2 -2.,'-k',linewidth=2.0)
	plt.plot([-np.sqrt(2),np.sqrt(2)],[0,0],'ok',markersize=8.)
	plt.xlabel('$x$',fontsize=16)
	plt.ylabel('$\dot{x}$',fontsize=16)
	# plt.title(r'$\dot{x} = \lambda + x^2,$ $\lambda = -2 $')
	T = np.linspace(0,5,500)
	plt.plot(T,np.zeros(T.shape),'-k')   # Graph x and c axes
	plt.plot(-T,np.zeros(T.shape),'-k')
	plt.plot(np.zeros(T.shape),T,'-k')
	plt.plot(np.zeros(T.shape),-T,'-k')
	plt.arrow(-.5, 0, -0.5, 0, head_width=0.1, head_length=0.1, fc='k', ec='k',linewidth=2.5)
	plt.arrow(1.15, 0, -0.5, 0, head_width=0.1, head_length=0.1, fc='k', ec='k',linewidth=2.5)
	plt.arrow(1.7, 0, .5, 0, head_width=0.1, head_length=0.1, fc='k', ec='k',linewidth=2.5)
	plt.arrow(-2.3, 0, .5, 0, head_width=0.1, head_length=0.1, fc='k', ec='k',linewidth=2.5)
	plt.axis([-2.5,2.5,-3,5])
	plt.savefig('SaddleNPhasePortrait.pdf')
	# plt.show()
	plt.clf()
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
	# plt.show()
	plt.clf()
	return


def Hysteresis():
	def F(x,lmbda):	return lmbda + x - x**3.
	
	plot_continuation_curve(np.linspace(-5,2,1400),-2,F,'-k',"Stable equilibria")
	plot_continuation_curve(np.linspace(5,-2,1400),2,F,'-k') 
	plot_continuation_curve(np.linspace(0,-2,700),0,F,'--k',"Unstable equilibria") 
	plot_continuation_curve(np.linspace(0,2,700),0,F,'--k') 
	x0, x1 = 1./np.sqrt(3), -1./np.sqrt(3)
	l0,l1 = x0**3 -x0,x1**3 -x1
	plt.plot([l0,l1],[x0,x1],'ok')
	plt.arrow(-.5, 0, 0, -.5, head_width=0.1, head_length=0.05, fc='k', ec='k',linewidth=1.5)
	plt.arrow(.5, 0, 0, .5, head_width=0.1, head_length=0.05, fc='k', ec='k',linewidth=1.5)
	plt.arrow(.4, 1.5, -.5, -.2, head_width=0.1, head_length=0.05, fc='k', ec='k',linewidth=1.5)
	plt.arrow(-.4, -1.5, .5, .2, head_width=0.1, head_length=0.05, fc='k', ec='k',linewidth=1.5)
	
	plt.legend(loc='best')
	plt.axis([-3,3,-3,3])
	plt.title("Hysteresis loop") # "\n Prototype equation: $x' = \lambda + x - x^3$")
	plt.xlabel('$\lambda$',fontsize=16)
	plt.savefig('HysteresisBifurcation.pdf')
	# plt.show()
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
	# plt.show()
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
	# plt.show()
	plt.clf()
	return



def problem4():
	r, k = .56, 8
	def f1(x,r,k):
		return r*(1.-x/k)
	
	def f2(x):
		return x/(1. + x**2.)
	
	x = np.linspace(0,10,100)
	plt.plot(x,f1(x,r,k),'-g',linewidth=2.,label=r'$r(1-x/k)$')
	plt.plot(x,f2(x),'-b',linewidth=2.,label=r'$x/(1+x^2)$')
	plt.axis([-.1,10,-.1,1.1])
	plt.legend(loc='best')
	plt.savefig('BudwormEquilibria.pdf')
	plt.show()
	plt.clf()
	
	def F(x,lambda_):	
		return x*(f1(x,r,lambda_)  - f2(x))
	
	from scipy.optimize import newton
	G = lambda x: F(x,9)
	soln1 = newton(G, .7, fprime=None, args=(), tol=1.0e-06, maxiter=80)
	soln2 = newton(G, 2, fprime=None, args=(), tol=1.0e-06, maxiter=80)
	soln3 = newton(G, 6, fprime=None, args=(), tol=1.0e-06, maxiter=80)
	
	
	print soln1, soln2, soln3
	# return 
	
	# plt.plot(x,F(x,9),'-k',linewidth=1.5)
	# plt.plot(x,np.zeros(x.shape),'-k',linewidth=1.5)
	# plt.show()
	l1, l2 = 0,15
	h1, h2 = -5,15
	N = 5000
	plot_continuation_curve(np.linspace(9,l2,N),soln1,F,'-k',plot_label='Stable Equilibria')
	plot_continuation_curve(np.linspace(9,2,N),soln1,F,'-k',plot_label=None) 
	plot_continuation_curve(np.linspace(9,l2,N),soln2,F,'--k',plot_label='Unstable Equilibria')
	plot_continuation_curve(np.linspace(9,2,N),soln2,F,'--k',plot_label=None)
	plot_continuation_curve(np.linspace(9,l2,N),soln3,F,'-k',plot_label=None)
	plot_continuation_curve(np.linspace(9,2,N),soln3,F,'-k',plot_label=None)
	
	C, X = EmbeddingAlg(np.linspace(9,12,N),soln1,F)
	plt.plot(C[-1],X[-1],'ok',markersize=7)
	C, X = EmbeddingAlg(np.linspace(9,2,N),soln2,F)
	plt.plot(C[-1],X[-1],'ok',markersize=7)
	C, X = EmbeddingAlg(np.linspace(9,2,N),soln3,F)
	plt.plot(C[-1],X[-1],'ok',markersize=7)
	
	# plot_continuation_curve(np.linspace(5,-2,1400),2,G,'-k') 
	# plot_continuation_curve(np.linspace(1,15,700),0,G,'--r',plot_label=None) 
	# x0, x1 = 1./np.sqrt(3), -1./np.sqrt(3)
	# l0,l1 = x0**3 -x0,x1**3 -x1
	# plt.plot([l0,l1],[x0,x1],'ok')
	# plt.arrow(-.5, 0, 0, -.5, head_width=0.1, head_length=0.05, fc='k', ec='k',linewidth=1.5)
	# plt.arrow(.5, 0, 0, .5, head_width=0.1, head_length=0.05, fc='k', ec='k',linewidth=1.5)
	# plt.arrow(.4, 1.5, -.5, -.2, head_width=0.1, head_length=0.05, fc='k', ec='k',linewidth=1.5)
	# plt.arrow(-.4, -1.5, .5, .2, head_width=0.1, head_length=0.05, fc='k', ec='k',linewidth=1.5)
	
	plt.legend(loc='best')
	plt.axis([l1,l2,h1,h2])
	# plt.title("Budworm Population\n" +  r"$x' = rx (1 - x/k ) - x/(1+x^2)$, $r=.56$")
	plt.xlabel('$k$',fontsize=16)
	plt.savefig('BudwormPopulation.pdf')
	plt.show()
	plt.clf()	
	return


# Another_bifurcation(-1)
# Another_bifurcation(-.2)
# Another_bifurcation(0)
# Another_bifurcation(.2)
# Another_bifurcation(1)
# SaddleNode_bifurcation()
# Hysteresis()
# Pitchfork_bifurcation()
# Transcritical_bifurcation()
problem4()



