import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

import shapes

#Prob 1

def polarZ(z):
	if(z == 0):
		return (0,0)
	else :
		a = z.real
		b = z.imag
		return( sp.hypot(a,b), sp.arctan(b/a))


#Prob 2

def fFunc(z):
	return (3*z - 4)/(z-2)
	
def gFunc(z):
	return z**2 - 2
def confMap(shape,mapfunc):
	shapemapped = [None]*len(shape)
	for i in range(0,len(shape)):
		shapemapped[i] = mapfunc(shape[i])
	
	plt.scatter(sp.real(shape),sp.imag(shape),color='r')
	plt.scatter(sp.real(shapemapped),sp.imag(shapemapped),color='b')
	plt . show ()

'''
Examples:

triangle = shapes.genRTriangle(1.,1.,0.,0.,100)
confMap(triangle,fFunc)

box = shapes.genBox(1.,1.,0.,0.,400)
confMap(box,gFunc)

circle = shapes.genCircle(1.,1.,0.,0.,400)
confMap(circle,gFunc)
'''

#Prob 3

def testHolo(func,a,b,r,tol):
	circle = shapes.genCircle(r,a,b,400)
	vals = func(circle)
	diff = max(vals) - min(vals)
	if(diff < tol ):
		return "differentiable"
	else:
		return "not differentiable {0}".format(diff)
#my solution apparently didn't work, it says that all the functions are not differentiable

print testHolo(lambda z: sp.real(z),1,1,0.01,1)
print testHolo(lambda z: sp.absolute(z),1,1,0.01,1)
print testHolo(lambda z: sp.conj(z),1,0,0.01,1)
print testHolo(lambda z: z*sp.real(z),0,0,0.01,1)
print testHolo(lambda z: z*sp.real(z),1,1,0.01,1)
print testHolo(lambda z: sp.exp(z),0,np.pi,0.01,1)

print testHolo(lambda z: sp.sin(sp.real(z))*sp.cosh(sp.imag(z))+
	sp.cos(sp.real(z))*sp.sinh(sp.imag(z))*1j ,0,np.pi,0.01,1)
print testHolo(lambda z: sp.sin(sp.real(z))*sp.cosh(sp.imag(z))
	-sp.cos(sp.real(z))*sp.sinh(sp.imag(z))*1j,0,np.pi,0.01,1)



