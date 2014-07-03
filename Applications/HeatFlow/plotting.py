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
	
	# # Generates the figures in the heat flow lab
	# view = [-.1, 1.1,-.1, .5]
	# plt.plot(x,u[:,0],'-k',linewidth=2.0)		# Initial Conditions
	# plt.plot(x,u[:,0],'ok',linewidth=2.0)
	# plt.axis(view)
	# plt.savefig('heatexercise1a.pdf')
	# plt.clf()
	# 
	# plt.plot(x,u[:,-1],'-k',linewidth=2.0)		# State at t = .2
	# plt.plot(x,u[:,-1],'ok',linewidth=2.0)
	# plt.axis(view)
	# plt.savefig('heatexercise1b.pdf')
	# plt.clf()
	
	Data = x,u[:,::1]
	Data = Data[0], Data[1].T[0:-1,:]
	time_steps = Data[1].shape[0]
	interval_length=50
	view = [-.1, 1.1,-.1, .5]
	math_animation(Data,time_steps,view,interval_length)
	return 


def Exercise2():
	x_subintervals, t_subintervals = 140.,70.
	x_interval,t_final = [-12,12], 1.
	init_conditions = np.maximum(1.-np.linspace(-12,12,x_subintervals+1)**2,0.)
	x,u = heatexplicit(init_conditions,x_subintervals,t_subintervals,x_interval,T=t_final,flag3d="off",nu=1.)
	
	# # Generates a figure in the heat flow lab
	# view = [-12, 12,-.1, 1.1]
	# plt.plot(x,u[:,0],'-k',linewidth=2.0,label="Initial State")	
	# plt.plot(x,u[:,-1],'--k',linewidth=2.0,label="State at time $t=1$.")		# State at t = .2
	# plt.xlabel("x",fontsize=16)
	# plt.legend(loc=1)
	# plt.axis(view)
	# plt.savefig('heatexercise2.pdf')
	# plt.show()
	# plt.clf()
	
	# Animation of results
	view = [-12, 12,-.1, 1.1]
	Data = x,u[:,::2]
	Data = Data[0], Data[1].T[0:-1,:]
	time_steps = Data[1].shape[0]
	interval_length=30
	math_animation(Data,time_steps,view,interval_length)
	return


def Exercise3():
	def graph(U,List,h):
		# # Error Chart 1: Plot approximate error for each solution
		# for j in range(0,len(List)-1): plt.plot(X[j],abs(U[-1][::2**(6-j)]-U[j]),'-k',linewidth=1.2)
		# for j in range(0,len(List)-1): plt.plot(X[j],abs(U[-1][::List[-1]/List[j]]-U[j]),'-k',linewidth=1.2)
		# plt.savefig("ApproximateError.pdf")
		# plt.clf()
		
		# # Error Chart 2: Create a log-log plot of the max error of each solution
		
		# MaxError = [max(abs(U[-1][::2**(6-j)]-U[j] )) for j in range(0,len(List)-1)]
		MaxError = [max(abs(U[-1][::List[-1]/List[j]]-U[j] )) for j in range(0,len(List)-1)]
		plt.loglog(h,MaxError,'-k',label='Error $E(h)$')
		plt.loglog(h,MaxError,'ko')
		plt.loglog(np.array(h),np.array(h)**2.,'-r',label='$h^2$')
		
		# plt.ylabel("",fontsize=16)
		plt.xlabel('$h$',fontsize=16)
		plt.legend(loc='best')
		h, MaxError = np.log(h)/np.log(10),np.log(MaxError)/np.log(10)
		print '\n'*2, "Approximate order of accuracy = ", (MaxError[-1]-MaxError[0])/(h[-1]-h[0])
		plt.savefig("MaximumError.pdf")
		plt.show()
		plt.clf()
		return
	
	
	# # This Graph is found in the lab
	# U = []
	# List = [20,40,80,160,320,640,1280,2560]
	# h = [1./item for item in List[:-1]]
	# for subintervals in List:
	# 	x_subintervals, t_subintervals = subintervals, subintervals
	# 	xinterval = [-12,12]
	# 	init_conditions = np.maximum(1.-np.linspace(-12,12,x_subintervals+1)**2,0.) 
	# 	x,u = heat_Crank_Nicolson(init_conditions, x_subintervals, t_subintervals,
	# 								x_interval=xinterval, T=1., flag3d="off", nu=1.)
	# 	U.append(u[:,-1])
	# 
	# graph(U,List,h)
	
	# x_subintervals, t_subintervals = 400.,500. 
	# xinterval = [-12,12]
	# init_conditions = np.maximum(1.-np.linspace(-12,12,x_subintervals+1)**2,0.) 
	# x,u = heat_Crank_Nicolson(init_conditions,x_subintervals,t_subintervals,x_interval=xinterval,T=1.,flag3d="off",nu=1.)
	# heat_Crank_Nicolson
	# print u
	# view = [-12, 12,-.1, 1.1]
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
	
	# Data = x,u[:,::6]
	# Data = Data[0], Data[1].T[0:-1,:]
	# time_steps = Data[1].shape[0]
	# interval_length=5
	# math_animation(Data,time_steps,view,interval_length)
	return



Exercise1()
Exercise2()
Exercise3()










