import numpy as np
import scipy as sp

#generate a circle of raduis r centered on a +bi
#of n points
def genCircle(r,a,b,n):
	circle = [None]*n
	thetas = np.arange(0,2*np.pi,2*np.pi/n)
	for i in range(0,len(thetas)):
		circle[i] = complex(np.cos(thetas[i])+a,np.sin(thetas[i])+b)
	return circle

#generate a box of width w and height h with the lower left corner on a + bi
#of n points
def genBox(w,h,a,b,n):
	box = [None]*n
	
	#the corners
	a1 = a
	a2 = a + w
	b1 = b
	b2 = b + h

	
	das = np.arange(a1,a2,w/(n/4.))
	dbs = np.arange(b1,b2,h/(n/4.))
	for i in range(0,n/4):
		box[i] = complex(das[i],b1)
		box[i+n/4] = complex(das[i],b2)
		box[i+n/2] = complex(a1,dbs[i])
		box[i+3*n/4] = complex(a2,dbs[i])
		
	return box
#generate a right triangle of base width w, height h, with the lower left corner on a + bi
#of n points *along each side
def genRTriangle(w,h,a,b,n):
	triangle = [None]*3*n

	#the corners
	a1 = a
	a2 = a + w
	b1 = b
	b2 = b + h

	das = np.arange(a1,a2,w/n)
	dbs = np.arange(b2,b1,-h/n)

	for i in range(0,n):
		triangle[i] = complex(das[i],b)
		triangle[i+n] = complex(a,dbs[i])
		triangle[i+2*n] = complex(das[i],dbs[i])
	return triangle
