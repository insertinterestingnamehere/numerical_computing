# import matplotlib
# matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
from solution import EmbeddingAlg

def plot_continuation_curve(c_list,Initial_Guess,F):
	C, X = EmbeddingAlg(c_list,Initial_Guess,F)
	plt.plot(C,X,'-g',linewidth=2.0)
	return


def plot_axes():
	T = np.linspace(0,5,500)
	plt.plot(T,np.zeros(T.shape),'-k')   # Graph x and c axes
	plt.plot(-T,np.zeros(T.shape),'-k')
	plt.plot(np.zeros(T.shape),T,'-k')
	plt.plot(np.zeros(T.shape),-T,'-k')
	plt.ylabel('x',fontsize=16)
	plt.axis([-5,5,-5,5])
	return


def SaddleNode_bifurcation():
	def F(x,c): return c + x**2
	
	plot_axes()
	plot_continuation_curve(np.linspace(-5,0,500),np.sqrt(5),F) 
	plot_continuation_curve(np.linspace(-5,0,500),-np.sqrt(5),F) 
	plt.title("Saddle-node bifurcation\n Prototype equation: $x' = c + x^2$")
	plt.xlabel('$c$',fontsize=16)
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


def Predicted_Weight(PAL,EI,Target_Weight,age,sex,H,BW,T):	
	from solution import fat_mass, compute_weight_curve, weight_odes
	F = fat_mass(BW,age,H,sex)
	L = BW-F
	# Fix activity level and determine Energy Intake to achieve Target weight of 145 lbs = 65.90909 kg
	
	init_FL = np.array([F,L])
	reltol, abstol = 1e-9,1e-8
	
	ode_f = lambda t,y: weight_odes(t,y,EI,PAL)
	ode_object = ode(ode_f).set_integrator('dopri5',atol=abstol,rtol=reltol) 
	ode_object.set_initial_value(init_FL, 0.) 
	y1 = sum(ode_object.integrate(T) )
	return y1-Target_Weight


def Weightloss_Continuation():
	age, sex =  38. , 'female'
	H, BW    =  1.73, 72.7 # in meters and kgs (about 160 lbs)
	T		 =  12*7.   # Time frame = 3 months   
	Target_Weight = 65.90909 # in kg (about 145 lbs)
	F = lambda EI, PAL: Predicted_Weight(PAL,EI,Target_Weight,age,sex,H,BW,T)
	
	c_list, Initial_Guess = np.linspace(1.4,2.4,10), 1283
	C, X = EmbeddingAlg(c_list,Initial_Guess,F)
	plt.plot(C,X,'-g',linewidth=2.5)
	plt.xlabel('Physical Activity Level',fontsize=14)
	plt.ylabel('Energy Intake',fontsize=14)
	plt.axis([1.3,2.5,1000,3500])
	plt.title("Options for a weightloss program for Jane for the next 3 months\n Goal: Lose 15 pounds",fontsize=16)
	plt.show(); plt.clf()
	return 




# Another_bifurcation(-1)
# SaddleNode_bifurcation()
# Hysteresis()
# Pitchfork_bifurcation()
# Transcritical_bifurcation()
Weightloss_Continuation()




