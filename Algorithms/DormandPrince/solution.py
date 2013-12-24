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

def compute_weight_curve(F,L,T,EI,PAL): 
	y0 = np.array([F,L])
	ode_f = lambda t,y: weight_step(t,y,EI,PAL)
	ode_object = ode(ode_f).set_integrator('dopri5',rtol=1e-6,atol=1e-8) 
	ode_object.set_initial_value( y0, 0.) 

	t = np.linspace(0.,T,151)
	y = np.zeros((len(t),len(y0))); 
	y[0,:] = y0

	for j in range(1,len(t)): y[j,:] = ode_object.integrate(t[j])  
	return t,y

def weight_step(t,y,EI,PAL):
	F, L = y[0], y[1]
	p = Forbes(F)
	EB = EnergyBalance(F,L,EI(t),PAL(t))
	return np.array([(1-p)*EB/c.rho_F,p*EB/c.rho_L]) 

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

def Forbes(F):
	C1 = c.C*c.rho_L/c.rho_F
	out = 1.*C1/(C1+F)
	return out

# Compare with Forbes and Mufflin
# Estimated initial body fat mass
# Jackson AS et al., Int J Obes Relat Metab Disord. 2002 Jun;26(6):789-96
# 

def fat_mass(BW,age,H,sex):
	BMI = BW/H**2.
	if sex=='male': 
		out = BW*(-103.91 + 37.31*np.log(BMI)+0.14*age)/100
	else:
		out = BW*(-102.01 + 39.96*np.log(BMI)+0.14*age)/100
	return out
	
def EnergyBalance(F,L,EI,PAL):
	p = Forbes(F)
	a1 = (1./PAL-c.beta_AT)*EI - K - c.gamma_F*F - c.gamma_L*L
	a2 = (1-p)*c.eta_F/c.rho_F + p*c.eta_L/c.rho_L+1./PAL
	EB = a1/a2
	return EB