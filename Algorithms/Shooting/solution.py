import numpy as np
from scipy.special import lambertw
from scipy.integrate import ode


#Global variables. May be updated in weightloss4
class c: pass

# 
# Fixed Constants
# 
c.rho_F		= 9400.   # 
c.rho_L		= 1800.   #
c.gamma_F	= 3.2     # 
c.gamma_L	= 22.     # 
c.eta_F		= 180.    # 
c.eta_L		= 230.    # 
c.C			= 10.4    # Forbes constant
c.beta_AT	= 0.14    # Adaptive Thermogenesis
c.beta_TEF	= 0.1     # Thermic Effect of Feeding

K 			= 0

# def getBW(F,L,T,EI,PAL): 
# 	t, y = compute_weight_curve(F,L,T,EI,PAL)
# 	out = np.sum(y[-1,:]) 
# 	return out


# def dBW(Fi,EIi,PALi,EIf,PALf): 
# # 
# # Given an intervention (EI,PAL), find the dBW achieved in equilibrium
# # 
# 	deltaEI = EIf - EIi
# 	psi = (1/PALf - 1/PALi)*EIi + (1/PALf-c.beta_AT)*deltaEI + c.gamma_F*Fi
# 	phi = c.gamma_F * Fi / (c.gamma_L * c.C)
# 	out = (psi - c.gamma_L*Fi + c.gamma_L*c.C*(c.gamma_L-c.gamma_F)/c.gamma_F * lambertw(phi*np.exp(psi/(c.C*c.gamma_L))))/c.gamma_L
# 	return out
# 
# 
# def dEI(Fi,deltaBW,EIi,PALi,PALf):
# # 
# # Given a desired BW, find the dEI needed to achieve that in equilibrium
# # 
# 	Ff = c.C*lambertw(np.exp(Fi/c.C)*np.exp(deltaBW/c.C)*Fi/c.C)
# 	chi = EIi/PALi + c.gamma_L*deltaBW+(c.gamma_F-c.gamma_L)*(Ff-Fi)
# 	out = (chi*PALf-EIi)/(1-c.beta_AT*PALf)
# 	return out
# 

# def generic_RMR(BW,age,H,sex): 
# # 
# # Mufflin equation
# # 
# 	if sex=='male': 
# 		out = 9.99*BW + 625*H - 4.92*age+5
# 	else:
# 		out = 9.99*BW + 625*H - 4.92*age-161
# 	return out

# def getK(F,L,EI,PAL,EB):
# 	if EB==0:
# 		p = 0
# 	else:
# 		p = Forbes(F)
# 	K = (1./PAL-c.beta_AT)*EI-c.gamma_L*L-c.gamma_F*F-((c.eta_F/c.rho_F)*(1-p)+(c.eta_L/c.rho_L)*p+1./PAL)*EB
# 	return K


def fat_mass(BW,age,H,sex):
	BMI = BW/H**2.
	if sex=='male': return BW*(-103.91 + 37.31*np.log(BMI)+0.14*age)/100
	else: return BW*(-102.01 + 39.96*np.log(BMI)+0.14*age)/100


def compute_weight_curve(F,L,T,EI,PAL): 
	y0 = np.array([F,L])
	ode_f = lambda t,y: weight_odes(t,y,EI(t),PAL(t))
	ode_object = ode(ode_f).set_integrator('dopri5',rtol=1e-6,atol=1e-8) 
	ode_object.set_initial_value( y0, 0.) 
	
	t = np.linspace(0.,T,151)
	y = np.zeros((len(t),len(y0))); 
	y[0,:] = y0
	
	for j in range(1,len(t)): y[j,:] = ode_object.integrate(t[j])  
	return t,y
	# return ode_object.integrate(T)  


def weight_odes(t,y,EI,PAL):
	F, L = y[0], y[1]
	p, EB = Forbes(F), EnergyBalance(F,L,EI,PAL)
	return np.array([(1-p)*EB/c.rho_F,p*EB/c.rho_L]) 


def EnergyBalance(F,L,EI,PAL):
	p = Forbes(F)
	a1 = (1./PAL-c.beta_AT)*EI - K - c.gamma_F*F - c.gamma_L*L
	a2 = (1-p)*c.eta_F/c.rho_F + p*c.eta_L/c.rho_L+1./PAL
	return a1/a2


def Forbes(F):
	C1 = c.C*c.rho_L/c.rho_F
	return 1.*C1/(C1+F)


# Compare with Forbes and Mufflin
# Estimated initial body fat mass
# Jackson AS et al., Int J Obes Relat Metab Disord. 2002 Jun;26(6):789-96
#

# 
# def Weightloss_Shooting():
# 	age, sex =  38. , 'female'
# 	H, BW    =  1.73, 72.7
# 	T		 =  12*7.   # Time frame = 3 months    
# 	
# 	PALf, EIf = 1.7, 2025
# 	def EI(t): return EIf
# 	
# 	def PAL(t): return PALf
# 	
# 	from solution import fat_mass, compute_weight_curve, weight_odes
# 	F = fat_mass(BW,age,H,sex)
# 	L = BW-F
# 	# Fix activity level and determine Energy Intake to achieve Target weight of 145 lbs = 65.90909 kg
# 	
# 	init_FL = np.array([F,L])
# 	EI0, EI1 = 2000, 1950					# Variable Energy Intake
# 	beta = 65.90909*2.2
# 	reltol, abstol = 1e-9,1e-8
# 	dim, iterate = 2, 15
# 	print '\nj = 1', '\nt = ', EI0
# 	for j in range(2,iterate):
# 		print '\nj = ', j, '\nt = ', EI1
# 		ode_f = lambda t,y: weight_odes(t,y,EI1,PAL(t))
# 		example1 = ode(ode_f).set_integrator('dopri5',atol=abstol,rtol=reltol) 
# 		example1.set_initial_value(init_FL, 0.) 
# 		y1 = 2.2*sum(example1.integrate(T) )
# 		
# 		if abs(y1-beta)<1e-8: 
# 			print '\n--Solution computed successfully--'
# 			break
# 		# Update
# 		ode_f = lambda t,y: weight_odes(t,y,EI0,PAL(t))
# 		example0 = ode(ode_f).set_integrator('dopri5',atol=abstol,rtol=reltol) 
# 		example0.set_initial_value(init_FL, 0.) 
# 		y0 = 2.2*sum(example0.integrate(T))
# 		
# 		EI2 = EI1 - (y1 - beta)*(EI1-EI0)/(y1- y0)
# 		EI0 = EI1
# 		EI1 = EI2
# 		
# 	# Here we plot the solution 
# 	ode_f = lambda t,y: weight_odes(t,y,EI1,PAL(t))
# 	example = ode(ode_f).set_integrator('dopri5',atol=abstol,rtol=reltol)
# 	example.set_initial_value(init_FL, 0.) 
# 	
# 	X = np.linspace(0.,T,801)
# 	Y = np.zeros((len(X),dim))
# 	Y[0,:] = init_FL
# 	
# 	for j in range(1,len(X)): Y[j,:] = example.integrate(X[j]) 
# 	
# 	print '\n|y(T) - Target Weight| = ', np.abs(beta-2.2*sum(Y[-1,:]) ),'\n'
# 	plt.plot(X,2.2*(Y[:,0]+Y[:,1]),'-k',linewidth=2.0)
# 	plt.axis([-1,T+1,0,170])
# 	plt.show(); plt.clf()
# 	return 
# 
# 
# Weightloss_Shooting()



# 
# Old Exercises: Example2, Exercise3, and Exercise4

# Good example of Newton's method.
# def Example2():
# 	a, b = 0., 3*np.pi/4.
# 	alpha, beta =  1., -(1.+3*np.sqrt(2))/2.
# 	dim, iterate = 4,10
# 	reltol, abstol,TOL = 1e-9,1e-8,1e-9
# 	t = (beta-alpha)/(b-a)  # Initial guess for the slope y'(1)
# 	def ode_f(x,y): 
# 		return np.array([y[1], -4.*y[0] - 9.*np.sin(x), 
# 						 y[3],  -4.*y[2]                  ])
# 	
# 	
# 	for j in range(1,iterate):
# 		print '\nj = ', j,'\nt = ', t
# 		
# 		example = ode(ode_f).set_integrator('dopri5',atol=abstol,rtol=reltol,nsteps=5000) 
# 		example.set_initial_value(np.array([alpha,t,0.,1.]),a) 
# 		X = example.integrate(b)
# 		y0, z0 = X[0],X[2]
# 		
# 		if abs(y0-beta)<TOL: 
# 			print '\n--Solution y computed successfully--',\
# 			'\n|y(b) - beta| = ',np.abs(beta-y0),'\n'
# 			break
# 		t = t - (y0 - beta)/z0   # Update guess y'(1) = t
# 		
# 	# Plot the computed solution vs. the analytic solution
# 	example = ode(ode_f).set_integrator('dopri5',atol=abstol,rtol=reltol)
# 	example.set_initial_value(np.array([alpha,t,0.,1.]),a) 
# 	T = np.linspace(a,b,401)
# 	Y = np.zeros((len(T),dim))
# 	Y[0,:] = np.array([alpha,t,0.,1.])
# 	
# 	for j in range(1,len(T)): 
# 		Y[j,:] = example.integrate(T[j])  
# 	plt.plot(T[::20],Y[::20,0],'*k')			# Numerical Solution
# 	Y2 = np.cos(2*T) + (1/2.)*np.sin(2*T) - 3.*np.sin(T)
# 	plt.plot(T,Y2,'-r')							# Exact Solution
# 	plt.show()	
# 	return


# def Exercise3(ya,va,nu,t0,t1): 
# 	a, b = 0., 120.
# 	beta =  0.
# 	# ya, va = 0.,35.
# 	# t0,t1 = np.pi/6., np.pi/7
# 	dim, iterate = 3,18
# 	reltol, abstol = 1e-9,1e-8
# 	
# 	# Initial_Conditions = np.array([z=0.,v=0.5,phi=t])
# 	# nu, g = 0., 9.8067
# 	g = 9.8067
# 	def ode_f(x,y): 
# 		# y = [z,v,phi]
# 		return np.array([np.tan(y[2]), -(g*np.sin(y[2]) + nu*y[1]**2.)/(y[1]*np.cos(y[2])), 
# 		-g/y[1]**2.])	
# 	
# 	
# 	# # Here we plot the solution 
# 	# example = ode(ode_f).set_integrator('dopri5',atol=abstol,rtol=reltol)
# 	# example.set_initial_value(np.array([ya,va,t0]),a) 
# 	# X = np.linspace(a,b,801)
# 	# Y = np.zeros((len(X),dim))
# 	# Y[0,:] = np.array([ya,va,t0])
# 	# 
# 	# for j in range(1,len(X)): 
# 	# 	Y[j,:] = example.integrate(X[j]) 
# 	# print '\n|y(b) - beta| = ',np.abs(beta-Y[-1,0]),'\n'
# 	# plt.plot(X,Y[:,0],'-k')
# 	# plt.show()
# 	# if True: return
# 	
# 	print '\nj = 1','\nt = ', t0
# 	for j in range(2,iterate):
# 		print '\nj = ', j,'\nt = ', t1
# 		example1 = ode(ode_f).set_integrator('dopri5',atol=abstol,rtol=reltol) 
# 		example1.set_initial_value(np.array([ya,va,t1]),a) 
# 		y1 = example1.integrate(b)[0]
# 		
# 		if abs(y1-beta)<1e-8: 
# 			print '\n--Solution y computed successfully--'
# 			break
# 		# Update
# 		example0 = ode(ode_f).set_integrator('dopri5',atol=abstol,rtol=reltol) 
# 		example0.set_initial_value(np.array([ya,va,t0]),a) 
# 		y0 = example0.integrate(b)[0]
# 		
# 		t2 = t1 - (y1 - beta)*(t1-t0)/(y1- y0)
# 		t0 = t1
# 		t1 = t2
# 		
# 	# Here we plot the solution 
# 	example = ode(ode_f).set_integrator('dopri5',atol=abstol,rtol=reltol)
# 	example.set_initial_value(np.array([ya,va,t1]),a) 
# 	X = np.linspace(a,b,801)
# 	Y = np.zeros((len(X),dim))
# 	Y[0,:] = np.array([ya,va,t1])
# 	
# 	for j in range(1,len(X)): 
# 		Y[j,:] = example.integrate(X[j]) 
# 	print '\n|y(b) - beta| = ',np.abs(beta-Y[-1,0]),'\n'
# 	# plt.plot(X,Y[:,0],'-k')
# 	return t2,X,Y[:,0]
# 
# 
# # y''=6y**2-6x**4-10, x in [1,3]
# # Solution is y=x**2 + x**(-2), y'(x) = 2x -2x^(-3), y'(1) = 0 
# # Shooting Algorithm: Converges for t0 in [-.1,.25]
# # If the interval is [1,2], convergence occurs for t0 in [-1.4,3.9]
# def Exercise4(t):
	# def ode_f(x,y): 
	# 	return np.array([y[1], 2. + 6.*y[0]**2.-12.*x**2.*y[0]+6.*x**4., 
	# 					 y[3],  12.*(y[0]-x**2.)*y[2]                  ])
	# 
	# a,b = 1.,2.
	# alpha, beta = 2., b**2 + b**(-2) # 4.+(1./4.)
	# reltol, abstol = 1e-9,1e-9
	# iterate = 15
	# # t = .01  # Initial guess for the slope y'(1)
	# 
	# # Plot the solution of the IVP with initial slope t
	# def ode_f2(x,y): 
	# 	return np.array([y[1], 2. + 6.*y[0]**2.-12.*x**2.*y[0]+6.*x**4.])
	# 
	# example = ode(ode_f2).set_integrator('dopri5',atol=abstol,rtol=reltol)
	# example.set_initial_value(np.array([alpha,t]),a) 
	# T = np.linspace(a,b,401)
	# dim=2
	# Y = np.zeros((len(T),dim))
	# Y[0,:] = np.array([alpha,t])
	# 
	# for j in range(1,len(T)): 
	# 	Y[j,:] = example.integrate(T[j])  
	# # plt.plot(T,Y[:,0],'-g')
	# # plt.show()
	# # plt.clf()
	# 
	# for j in range(1,iterate):
	# 	print '\nj = ', j,'\nt = ', t
	# 	
	# 	example1 = ode(ode_f).set_integrator('dopri5',atol=abstol,rtol=reltol,nsteps=5000) 
	# 	example1.set_initial_value(np.array([alpha,t,0.,1.]),a) 
	# 	X = example1.integrate(b)
	# 	y0, z0 = X[0],X[2]
	# 	
	# 	if abs(y0-beta)<1e-9: 
	# 		print '\n--Solution y computed successfully--',\
	# 		'\n|y(b) - beta| = ',np.abs(beta-y0),'\n'
	# 		break
	# 	t = t - (y0 - beta)/z0   # Update guess y'(1) = t
	# 	
	# # Plot the computed solution vs. the analytic solution
	# example = ode(ode_f).set_integrator('dopri5',atol=abstol,rtol=reltol)
	# example.set_initial_value(np.array([alpha,t,0.,1.]),a) 
	# T = np.linspace(a,b,401)
	# dim=4
	# Y = np.zeros((len(T),dim))
	# Y[0,:] = np.array([alpha,t,0.,1.])
	# 
	# for j in range(1,len(T)): 
	# 	Y[j,:] = example.integrate(T[j])  
	# plt.plot(T[::10],Y[::10,0],'*k')
	# Y2 = T**2+T**(-2)
	# plt.plot(T,Y2,'-r')
	# plt.show()	
	# return


# Example2()



# Exercise4(.2)


# phi1,X,Y = Exercise3(0.,35., 0.,np.pi/3.5,np.pi/3.)
# plt.plot(X,Y,'-k',linewidth=2.0)
# phi2,X,Y = Exercise3(0,35.,0.,np.pi/6.,np.pi/6.5)
# plt.plot(X,Y,'-k',linewidth=2.0)
# print '\n'*4,"Angle giving the furthest distance: "
# print 'pi/4 =   ', np.pi/4.
# print "\nSolutions to BVP:"
# print "phi(0) = ", phi1
# print "         ", phi2
# plt.axis([-5,125,-5,45])
# plt.show()
