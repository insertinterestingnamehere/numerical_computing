
import scipy as sp
import scipy.linalg as la

def mcUnit(func,numPoints,dims):
	points = sp.rand(numPoints,dims)
	points = 2*(points-.5)
	total = sp.sum(sp.apply_along_axis(func,1,points))
	return float(total)/numPoints
		
def mcUnitConvergeEst(func,dims,minPoints,maxPoints,numTestPoints,testRuns):
	#Couldn't get this to work, spits out an answer near 0
	testPoints=sp.around(sp.linspace(minPoints,maxPoints,numTestPoints))
	error = sp.zeros(sp.size(testPoints))
	area = sp.zeros(testRuns)

	for i in range(0,numTestPoints):
		for k in range(0,testRuns):
			area[k]=mcUnit(func,testPoints[i],dims)
		error[i] = sp.mean(sp.absolute(area-sp.pi))

	estimate = la.lstsq(sp.vstack((sp.log(testPoints),sp.ones(sp.size(testPoints)))).T,sp.log(error))
	return estimate

def flawed_mcUnit(func,numPoints,dims):
	points = sp.rand(numPoints,dims)
	points =(2-.05)*(points)-0.95
	total = sp.sum(sp.apply_along_axis(func,1,points))
	return float(total)/numPoints	
	
if __name__ == "__main__":
	mcUnit(lambda x:sp.sin(x[0])*x[1]**5-x[1]**3+x[2]*x[3]+x[1]*x[2]**3,10000,4)
	mcUnitConvergeEst(lambda x:sp.sin(x[0])-x[1]**3+x[2]*x[3]+x[1]*x[2]**3,4,1000,10000,10,10)
	mcUnit.flawed_mcUnit(lambda x:sp.sin(x[0])-x[1]**3+x[2]*x[3]+x[1]*x[2]**3,10000,4)

