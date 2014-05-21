from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from solution import heatexplicit, math_animation, heat_Crank_Nicolson


def examplecode():
	pass


def Exercise1():
	x_subintervals, t_subintervals = 6,10
	x_interval,t_final = [0,1], .4
	init_conditions = 2.*np.maximum(1./5 - np.abs(np.linspace(0,1,x_subintervals+1)-1./2),0.)
	x,u = heatexplicit(init_conditions,x_subintervals,t_subintervals,x_interval,T=t_final,flag3d="on",nu=.05)
	
	view = [-.1, 1.1,-.1, .5]
	plt.plot(x,u[:,0],'-k',linewidth=2.0)		# Initial Conditions
	plt.plot(x,u[:,0],'ok',linewidth=2.0)
	plt.axis(view)
	plt.savefig('heatexercise1a.pdf')
	plt.clf()
	
	plt.plot(x,u[:,-1],'-k',linewidth=2.0)		# State at t = .2
	plt.plot(x,u[:,-1],'ok',linewidth=2.0)
	plt.axis(view)
	plt.savefig('heatexercise1b.pdf')
	plt.clf()
	
	# Data = x,u[:,::1]
	# Data = Data[0], Data[1].T[0:-1,:]
	# time_steps = Data[1].shape[0]
	# interval_length=10
	# math_animation(Data,time_steps,view,interval_length)
	return 


def Exercise2():
	x_subintervals, t_subintervals = 140.,70.
	x_interval,t_final = [-12,12], 1.
	init_conditions = np.maximum(1.-np.linspace(-12,12,x_subintervals+1)**2,0.)
	x,u = heatexplicit(init_conditions,x_subintervals,t_subintervals,x_interval,T=t_final,flag3d="off",nu=1.)
	view = [-12, 12,-.1, 1.1]
	plt.plot(x,u[:,0],'-k',linewidth=2.0,label="Initial State")	
	plt.plot(x,u[:,-1],'--k',linewidth=2.0,label="State at time $t=1$.")		# State at t = .2
	plt.xlabel("x",fontsize=16)
	plt.legend(loc=1)
	plt.axis(view)
	plt.savefig('heatexercise2.pdf')
	# plt.clf()
	plt.show()
	
	# # Animation of results
	# Data = x,u[:,::2]
	# Data = Data[0], Data[1].T[0:-1,:]
	# time_steps = Data[1].shape[0]
	# interval_length=10
	# math_animation(Data,time_steps,view,interval_length)
	return


def Exercise3():
	x_subintervals, t_subintervals = 400.,500. 
	x_interval,t_final = [-12,12], 1. 
	init_conditions = np.maximum(1.-np.linspace(-12,12,x_subintervals+1)**2,0.) 
	x,u = heat_Crank_Nicolson(init_conditions,x_subintervals,t_subintervals,x_interval=[-10,10],T=1.,flag3d="off",nu=1.)
	# heat_Crank_Nicolson
	# print u
	view = [-12, 12,-.1, 1.1]
	# plt.plot(x,u[:,0],'-k',linewidth=2.0)		# Initial Conditions
	# plt.axis(view)
	# # plt.savefig('heatexercise3a.pdf')
	# # plt.clf()
	# # 
	# plt.plot(x,u[:,-1],'-k',linewidth=2.0)		# State at t = .2
	# # plt.axis(view)
	# # plt.savefig('heatexercise3b.pdf')
	# # plt.clf()
	# plt.show()
	
	Data = x,u[:,::3]
	Data = Data[0], Data[1].T[0:-1,:]
	time_steps = Data[1].shape[0]
	interval_length=10
	math_animation(Data,time_steps,view,interval_length)
	return



# Exercise1()
Exercise2()
# Exercise3()










