import numpy as np
import matplotlib.pyplot as plt





def rivercrossing():
	x0 = np.linspace(-1,1,200)
	
	def r(x):
		return -.75*np.sqrt(1.-x**2.)
	
	
	y0 = r(x0)
	x1 = .5*(x0 + 1)
	
	def rp(x):
		return -9./16.*x/r(x)
	
	def a(x):
		return np.sqrt(1.-r(x)**2.)**(-.5)
	
	def ap():
		return r(x)*rp(x)*alpha(x)**3.
	
	
	def func(x):
		a_ = a(x)
		ap_ = ap(x)
		r_ = r(x)
		rp_ = rp(x)
		yp = D.dot(y)
		ypp = M.dot(y) # M = D.dot(D)
		out = ( (3.*ap_*yp + a_*ypp)*(1. + a_**2.*yp**2.) - 
				(a_**2.*yp)*(ap_*yp**2. + a_**2.*yp*ypp) - 
				(2./a_*ap_*r_ + rp_)*(1. + a_**2.*yp**2.)**(3./2)
		)
		return out
	
	
	plt.plot(x1,y0,'-k')
	plt.axis([-.1,1.1,-.8,.05])
	plt.show()
	
	

if __name__=="__main__":
	rivercrossing()














