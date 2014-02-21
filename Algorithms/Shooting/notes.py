import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
# def ode_f(x,y): 
	# return np.array([y[1] , -4.*y[0] - 9.*np.sin(x)])
	# return np.array([y[1] , 6.*y[0]**2. - 6.*x**4.-10.])
	# return np.array([y[1] , 6.*y[0]**2. + 6.*x**4.+2.-12.*x**2.*y[0]])


# a,b = .5, 3.
# init_conds = np.array([4.25,-15.])
# abstol, reltol = 1e-9, 1e-9
# 
# example = ode(ode_f).set_integrator('dopri5',atol=abstol,rtol=reltol)
# example.set_initial_value(init_conds,a)
# 
# dim, t = 2, np.linspace(a,b,401)
# Y = np.zeros((len(t),dim))
# 
# Y[0,:] = init_conds
# for j in range(1,len(t)): 
# 	Y[j,:] = example.integrate(t[j])  
# print 'y(3*pi) = ', Y[-1,0]
# plt.plot(t,Y[:,0],'-k')
# plt.axis([.5,3.,2,10])
# plt.show()


x = np.linspace(0.,(3./4.)*np.pi,200)
y = np.cos(2.*x)+ (1./2.)*np.sin(2.*x) -3.*np.sin(x)
plt.plot(x,y)
# plt.axis([0.,1.,2,10])
plt.show()

# j=0
# while j<500:
#     j=j+1
#     if j>10:
#         break
#     print j
