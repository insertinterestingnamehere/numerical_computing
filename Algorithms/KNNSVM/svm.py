import scipy as sp
from scipy import linalg as la
from cvxopt import matrix,solvers
import math

class SVM(object):

	def __init__(self,data,target):
		self.data = data
		self.target = target.astype(float)
		self.n_samples = len(target)

	def setKernel(self,type="polynomial"):
		if type == "linear":
			self.kernel = lambda x,y: sp.dot(x,y)
		elif type == "polynomial":
			a = float(raw_input("a: "))
			d = int(raw_input("d: "))
			self.kernel = lambda x,y: (sp.dot(x,y) + a)**d
		elif type == "rbf":
			gam = float(raw_input("Gamma: "))
			while gam <= 0:
				print "Gamma must be positive.\n"
				gam = float(raw_input("Gamma: "))
			self.kernel = lambda x,y: sp.exp(-gam*la.norm(x - y)**2)
		elif type == "sigmoid":
			r = float(raw_input("r: "))
			self.kernel = lambda x,y: math.tanh(sp.dot(x,y) + r)
		else:
			print "Type either not provided or not understood. Try again"
	
	def fit(self):
		K = sp.zeros((self.n_samples,self.n_samples))
		for i in xrange(self.n_samples):
			for j in xrange(self.n_samples):
				K[i,j] = self.kernel(self.data[i,:],self.data[j,:])

		P = matrix(sp.outer(self.target,self.target)*K)
		q = matrix(sp.ones(self.n_samples) * -1)
		A = matrix(self.target,(1,self.n_samples))
		b = matrix(0.0)
		G = matrix(sp.diag(sp.ones(self.n_samples)*-1))
		h = matrix(sp.zeros(self.n_samples))

		sol = solvers.qp(P,q,G,h,A,b)
		self.a = sp.ravel(sol['x'])

	def predict(self,x):
		if x.ndim == 1:
			temp = int(sp.dot(self.a*self.target,sp.array([self.kernel(x,self.data[i,:]) for i in xrange(self.n_samples)])) > 0)
			if temp:
				return 1
			else:
				return -1
		else:
			temps = sp.array([int(sp.dot(self.a*self.target,sp.array([self.kernel(x[j,:],self.data[i,:]) for i in xrange(self.n_samples)])) > 0) for j in xrange(x.shape[0])])
			inds = temps == 0
			if sum(inds) == 0:
				return temps
			else:
				temps[inds] = -1*sp.ones(sum(inds)).astype(int)
				return temps
