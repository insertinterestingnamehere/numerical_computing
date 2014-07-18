import numpy as np
import matplotlib.pyplot as plt


def example1():
	import numpy as np
	import matplotlib.pyplot as plt
	
	# (x,t) \in [0,1] \times [0,t_final]
	t_final = 1.2
	wind = 1.2 # Speed of advection or wind
	
	J = 600 # number of x subintervals
	tsteps = int( J*1.5)
	delta_x, delta_t = 1./(J), t_final/tsteps
	message = ("CFL Stability Condition: Need wind*delta_t/delta_x = "
				+	str(wind*delta_t/delta_x) +"<= 1.")
	assert (wind*delta_t/delta_x) <= 1., message 
	
	x_arr = np.linspace(0.,1.,J+1)[:-1]
	u, arr = np.exp(-(x_arr-.3)**2./.005), np.where( (.6 < x_arr) & (x_arr < .7 ) )
	u[arr] += 1.
	
	upwind = np.empty((tsteps+1,J))
	LaxWendroff = np.empty((tsteps+1,J))
	MinMod = np.empty((tsteps+1,J))
	SuperBee = np.empty((tsteps+1,J))
	upwind[0,:], LaxWendroff[0,:] = u[:], u[:]
	MinMod[0,:], SuperBee[0,:] = u[:], u[:]
	for j in xrange(tsteps): 
		# SuperBee Method
		delta = np.roll(MinMod[j,:],0) - np.roll(MinMod[j,:],1)
		
		slope1 = .5*(np.sign(delta) + np.sign(np.roll(delta,-1)))*(
					np.minimum( 2.*np.abs(delta), 
								np.abs(np.roll(delta,-1) ) ) )/delta_x
		
		slope2 = .5*( np.sign(delta) + np.sign(np.roll(delta,-1)) )*(
					np.minimum( np.abs(delta), 
								2.*np.abs(np.roll(delta,-1) ) ))/delta_x
		
		slope = .5*(np.sign(slope1) + np.sign(slope2))*(
					np.maximum( np.abs( slope1), 
								np.abs( slope2 ) ) )
								
		SuperBeeFlux = wind*(SuperBee[j,:]+.5*(1.-wind*delta_t/delta_x)*delta_x*slope)
		SuperBee[j+1,:] = SuperBee[j,:] - (delta_t/delta_x)*( 
							np.roll(SuperBeeFlux,0) - np.roll(SuperBeeFlux,1) 
															 )
		
		
		# Min Mod Method
		delta = np.roll(MinMod[j,:],0) - np.roll(MinMod[j,:],1)
		slope = .5*(np.sign(delta) + np.sign(np.roll(delta,-1)))*(
					np.minimum( np.abs(delta), 
								np.abs(np.roll(delta,-1) ) )/delta_x 
												)
		MinModFlux = wind*(MinMod[j,:]+.5*(1.-wind*delta_t/delta_x)*delta_x*slope)
		MinMod[j+1,:] = MinMod[j,:] - (delta_t/delta_x)*( 
							np.roll(MinModFlux,0) - np.roll(MinModFlux,1) 
															 )	
		# Upwind Method 
		upwind[j+1,:] = (upwind[j,:] - 
			wind*delta_t/delta_x*( upwind[j,:] - np.roll(upwind[j,:],1) ) )
		# Lax Wendroff Method
		LaxWendroff[j+1,:] = (LaxWendroff[j,:] - 
		.5*wind*delta_t/delta_x*( np.roll(LaxWendroff[j,:],-1) - 
								  np.roll(LaxWendroff[j,:],1) ) + 
		.5*(wind*delta_t/delta_x)**2.*( np.roll(LaxWendroff[j,:],1) - 
											 2.*LaxWendroff[j,:] +
										np.roll(LaxWendroff[j,:],-1) ) )
	# Compare analytic and numerical solutions
	dt = wind*t_final # distance traveled
	forward = int((dt - np.floor(dt))*x_arr.shape[0])
	
	plt.plot(x_arr ,np.roll(u,forward),'-k',linewidth=2.,label='Analytic Solution')
	plt.plot(x_arr , upwind[-1,:],'-r',linewidth=2.,label='Upwind Method')
	plt.plot(x_arr , LaxWendroff[-1,:],'-g',linewidth=2.,label='Lax Wendroff Method')
	plt.plot(x_arr , MinMod[-1,:],'-b',linewidth=2.,label='Min Mod Method')
	plt.plot(x_arr , SuperBee[-1,:],'-c',linewidth=2.,label='SuperBee Method')
	
	plt.legend(loc='best')
	# plt.axis([0, 1,-.1,1.1])
	plt.show()
	return 











example1()




